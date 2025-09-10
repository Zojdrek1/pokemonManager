"""
price_update.py — drop-in module for your Streamlit Pokémon Manager

What you get:
- Update button fetches BOTH Raw and PSA 10 prices for each card.
- Each card's prices are written back to the CSV *immediately* (atomic write),
  so if the run is interrupted, you keep everything fetched so far.
- Maintains your semicolon (;) CSV with header:
    ID;lookupID;Name;Set;Type;Quantity;Raw Price;Raw Total
  and adds these columns if missing:
    PSA10 Price;PSA10 Total

How to use:
1) Save this file as price_update.py in your project.
2) In your Streamlit app, do: from price_update import update_all_prices
3) Wire your sidebar button to call update_all_prices(csv_path, base_url),
   where base_url builds your price pages (example for PriceCharting shown).
"""
from __future__ import annotations
import os
import time
import tempfile
import shutil
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------- Config --------------
REQUEST_TIMEOUT = 20
RETRY_COUNT = 2
RETRY_SLEEP = 1.5
SLEEP_BETWEEN_REQUESTS = 0.6  # be a polite scraper

RAW_COL = "Raw Price"
RAW_TOTAL_COL = "Raw Total"
PSA10_COL = "PSA10 Price"
PSA10_TOTAL_COL = "PSA10 Total"
QTY_COL = "Quantity"
LOOKUP_COL = "lookupID"

# -------------- Utilities --------------

def _atomic_write(path: str, data: str, encoding: str = "utf-8") -> None:
    """Write text atomically to avoid partial/corrupted files on crashes."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, encoding=encoding) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)


def _save_df_semicolon(df: pd.DataFrame, csv_path: str) -> None:
    csv_text = df.to_csv(index=False, sep=";", line_terminator="\n")
    _atomic_write(csv_path, csv_text)


# -------------- Price fetchers --------------
@dataclass
class Prices:
    raw: Optional[float]
    psa10: Optional[float]


def _parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    # Keep digits and dot
    cleaned = "".join(ch for ch in text if (ch.isdigit() or ch == "."))
    if cleaned.count(".") > 1:
        # crude fix for things like 1.234.56
        parts = cleaned.split(".")
        cleaned = parts[0] + "." + "".join(parts[1:])
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def fetch_from_pricecharting(card_slug: str, base_url: str) -> Prices:
    """
    Fetch prices from a PriceCharting-style page.
    - card_slug is your existing lookupID, e.g. "pokemon-scarlet-&-violet/alomomola-48"
    - base_url example: "https://www.pricecharting.com/game/" (trailing slash required)

    Returns Prices(raw, psa10). Missing PSA10 returns None.
    """
    url = base_url.rstrip("/") + "/" + card_slug
    last_exc = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PokemonManager/1.0)"
            })
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            soup = BeautifulSoup(r.text, "html.parser")

            # Heuristics for PriceCharting layout
            raw_val = None
            psa10_val = None

            # Try common labels first
            # e.g. <td>Loose Price</td> £1.23, <td>Graded 10</td> £45.67
            table = soup.find("table", class_="price js-price-table") or soup
            for row in table.find_all(["tr", "div"]):
                txt = row.get_text(" ", strip=True).lower()
                if "loose price" in txt or "raw" in txt:
                    raw_val = _parse_money(txt)
                if "graded 10" in txt or "psa 10" in txt or "psa10" in txt:
                    val = _parse_money(txt)
                    if val is not None:
                        psa10_val = val

            # Fallbacks: look for microdata or meta tags
            if raw_val is None:
                meta = soup.find("meta", {"itemprop": "price"})
                if meta and meta.get("content"):
                    raw_val = _parse_money(meta["content"])  # best-effort

            return Prices(raw=raw_val, psa10=psa10_val)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_SLEEP)
            else:
                break
    # If completely failed, return Nones; caller decides how to handle
    return Prices(raw=None, psa10=None)


# -------------- Main update routine --------------

def ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if RAW_COL not in df.columns:
        df[RAW_COL] = 0.0
    if RAW_TOTAL_COL not in df.columns:
        df[RAW_TOTAL_COL] = 0.0
    if PSA10_COL not in df.columns:
        df[PSA10_COL] = 0.0
    if PSA10_TOTAL_COL not in df.columns:
        df[PSA10_TOTAL_COL] = 0.0
    return df


def update_all_prices(
    csv_path: str,
    base_url: str = "https://www.pricecharting.com/game",
    save_every: int = 1,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    fetcher: Callable[[str, str], Prices] = fetch_from_pricecharting,
) -> None:
    """
    Update raw + PSA10 prices for each row in the CSV, saving as we go.

    Parameters
    ----------
    csv_path : str
        Path to your semicolon CSV.
    base_url : str
        Base URL used by the fetcher.
    save_every : int
        Write the CSV to disk after this many updated rows.
    progress_cb : callable(idx, total, message)
        Optional callback (Streamlit/CLI) to report progress.
    fetcher : function
        Function taking (lookup_id, base_url) -> Prices.
    """
    df = pd.read_csv(csv_path, sep=";")
    df = ensure_price_columns(df)

    updated = 0
    total = len(df)

    for idx, row in df.iterrows():
        lookup_id = str(row.get(LOOKUP_COL, "")).strip()
        qty = float(row.get(QTY_COL, 0) or 0)
        if not lookup_id:
            if progress_cb:
                progress_cb(idx + 1, total, f"Skipping row {idx+1}: missing lookupID")
            continue

        prices = fetcher(lookup_id, base_url)
        raw_price = prices.raw if prices.raw is not None else row.get(RAW_COL, 0) or 0
        psa10_price = prices.psa10 if prices.psa10 is not None else 0  # 0 means unavailable

        df.at[idx, RAW_COL] = round(float(raw_price or 0), 2)
        df.at[idx, RAW_TOTAL_COL] = round(df.at[idx, RAW_COL] * qty, 2)
        df.at[idx, PSA10_COL] = round(float(psa10_price or 0), 2)
        df.at[idx, PSA10_TOTAL_COL] = round(df.at[idx, PSA10_COL] * qty, 2)

        updated += 1
        if progress_cb:
            progress_cb(idx + 1, total, f"Updated {lookup_id}")

        # Persist progress incrementally
        if updated % save_every == 0:
            _save_df_semicolon(df, csv_path)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Final save in case the total wasn't divisible by save_every
    _save_df_semicolon(df, csv_path)


# -------------- Streamlit wiring (example) --------------
"""
In your app.py, wire the sidebar button like this:

import streamlit as st
from price_update import update_all_prices

CSV_PATH = "cards.csv"  # or st.file_uploader/saved path
BASE_URL = "https://www.pricecharting.com/game"

st.sidebar.header("Prices")
if st.sidebar.button("Update Prices (Raw + PSA 10)"):
    status = st.empty()
    bar = st.progress(0)

    def cb(done, total, msg):
        bar.progress(int(done/total*100))
        status.write(f"{done}/{total} · {msg}")

    update_all_prices(CSV_PATH, BASE_URL, save_every=1, progress_cb=cb)
    status.write("Done! Saved to CSV as we went.")
"""
