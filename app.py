import re
# app.py
import os
import base64
import urllib.parse, re, io, json, base64, requests, pandas as pd
from urllib.parse import quote_plus, urljoin, urlparse
from bs4 import BeautifulSoup
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, DataReturnMode
from inspect import signature
import streamlit as st


# ---------- Query-param compatibility helpers (Streamlit >=1.30 and older) ----------
# ---------- Unified CSV save helper ----------
def save_quantities_to_csv(live_csv_path: str, edited_df: pd.DataFrame) -> int:
    """
    Save quantities from edited_df into the live CSV at live_csv_path.
    Matching priority:
      1) lookupID (exact, case-insensitive)
      2) ID
      3) composite key: Set|Name|ID|Type
    Returns number of updated rows.
    """
    # Load live CSV
    live = pd.read_csv(live_csv_path, delimiter=";")

    # Ensure columns exist
    needed_cols = ["ID","lookupID","Set","Name","Type","Quantity","Raw Price","Raw Total"]
    for col in needed_cols:
        if col not in live.columns:
            live[col] = 0 if col in ("Quantity","Raw Price","Raw Total") else ""

    # Normalize types
    live["ID"] = live["ID"].astype(str)
    if "lookupID" in live.columns: live["lookupID"] = live["lookupID"].astype(str)
    for col in ("Set","Name","Type"):
        if col in live.columns:
            live[col] = live[col].astype(str)
    live["Quantity"]  = pd.to_numeric(live["Quantity"], errors="coerce").fillna(0).astype(int)
    live["Raw Price"] = pd.to_numeric(live["Raw Price"], errors="coerce").fillna(0.0)
    if "Raw Total" in live.columns:
        live["Raw Total"] = pd.to_numeric(live["Raw Total"], errors="coerce").fillna(0.0)

    # Edited DF normalization
    if edited_df is None or len(edited_df)==0:
        return 0
    ed = edited_df.copy()
    for col in ("ID","lookupID","Set","Name","Type"):
        if col in ed.columns:
            ed[col] = ed[col].astype(str)
        else:
            ed[col] = ""
    ed["Quantity"] = pd.to_numeric(ed.get("Quantity", 0), errors="coerce").fillna(0).astype(int)

    updated_rows = 0

    # 1) lookupID match (case-insensitive exact)
    if "lookupID" in live.columns and ed["lookupID"].astype(str).str.strip().ne("").any():
        live_lu = live["lookupID"].astype(str).str.strip().str.lower()
        ed_lu   = ed["lookupID"].astype(str).str.strip().str.lower()
        lu_to_qty = dict(zip(ed_lu, ed["Quantity"].astype(int)))
        for idx, lu in live_lu.items():
            if lu in lu_to_qty:
                new_q = int(lu_to_qty[lu])
                if int(live.at[idx,"Quantity"]) != new_q:
                    live.at[idx,"Quantity"] = new_q
                    try:
                        rp = float(live.at[idx,"Raw Price"])
                        live.at[idx,"Raw Total"] = round(new_q * rp, 2)
                    except Exception:
                        pass
                    updated_rows += 1

    # 2) ID match (exact, string compare)
    live_id = live["ID"].astype(str)
    ed_id   = ed["ID"].astype(str)
    id_to_qty = dict(zip(ed_id, ed["Quantity"].astype(int)))
    for idx, idv in live_id.items():
        if idv in id_to_qty:
            new_q = int(id_to_qty[idv])
            if int(live.at[idx,"Quantity"]) != new_q:
                live.at[idx,"Quantity"] = new_q
                try:
                    rp = float(live.at[idx,"Raw Price"])
                    live.at[idx,"Raw Total"] = round(new_q * rp, 2)
                except Exception:
                    pass
                updated_rows += 1

    # 3) Composite fallback
    def make_key(df):
        def col(c): return df[c].astype(str) if c in df.columns else ""
        return (col("Set") + "|" + col("Name") + "|" + col("ID") + "|" + col("Type")).str.lower().str.strip()

    live_key = make_key(live)
    ed_key   = make_key(ed)
    ed_map   = dict(zip(ed_key, ed["Quantity"].astype(int)))
    for idx, key in live_key.items():
        if key in ed_map:
            new_q = int(ed_map[key])
            if int(live.at[idx,"Quantity"]) != new_q:
                live.at[idx,"Quantity"] = new_q
                try:
                    rp = float(live.at[idx,"Raw Price"])
                    live.at[idx,"Raw Total"] = round(new_q * rp, 2)
                except Exception:
                    pass
                updated_rows += 1

    if updated_rows > 0:
        live.to_csv(live_csv_path, sep=";", index=False)

    return updated_rows

def _get_query_params():
    try:
        return dict(st.query_params)  # New API (1.30+)
    except Exception:
        try:
            return dict(st.experimental_get_query_params())  # Legacy
        except Exception:
            return {}

def _set_query_param(key, value):
    try:
        st.query_params[key] = value
    except Exception:
        try:
            cur = dict(st.experimental_get_query_params())
            cur[key] = value
            st.experimental_set_query_params(**cur)
        except Exception:
            pass

def _clear_query_params(keys):
    try:
        for k in keys:
            try:
                del st.query_params[k]
            except Exception:
                pass
    except Exception:
        try:
            cur = dict(st.experimental_get_query_params())
            for k in keys:
                cur.pop(k, None)
            st.experimental_set_query_params(**cur)
        except Exception:
            pass

# --- Session state defaults ---
if 'saved_rate' not in st.session_state:
    st.session_state.saved_rate = 0.8  # sensible default


# ---------------- Paths & defaults ----------------
CSV_PATH = "db.csv"
CONFIG_PATH = "config.json"
PC_BASE = "https://www.pricecharting.com"
IMG_CACHE_DIR = os.path.join("cache", "pc_images")
os.makedirs(IMG_CACHE_DIR, exist_ok=True)
DEFAULT_RATE = 0.74

# ---------------- Config ----------------
def load_rate():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return float(json.load(f).get("usd_to_gbp", DEFAULT_RATE))
    except Exception:
        pass
    return DEFAULT_RATE

def save_rate(rate: float):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"usd_to_gbp": float(rate)}, f, indent=2)
    except Exception:
        pass

# ---------------- Helpers ----------------
LANG_PREFIXES = ["japanese","english","german","french","spanish","italian","korean","chinese","portuguese"]

def parse_lookup_id(value: str) -> str:
    value = (value or "").strip()
    m = re.search(r"pricecharting\.com/game/([^?\s#]+)", value)
    return m.group(1) if m else value

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip())
    return s.strip("-")[:150] or "img"

def cached_image_path(lookup_id: str) -> str:
    return os.path.join(IMG_CACHE_DIR, sanitize_filename(lookup_id) + ".webp")

def is_valid_image(path: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 200: return False
        with Image.open(path) as im: im.verify()
        return True
    except Exception:
        return False

def path_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f: b = f.read()
        return "data:image/webp;base64," + base64.b64encode(b).decode("ascii")
    except Exception:
        return ""

def absolutize_url(url: str, base: str = PC_BASE) -> str:
    if not url: return ""
    if url.startswith("//"): return "https:" + url
    if url.startswith("/"):  return urljoin(base, url)
    if urlparse(url).scheme in ("http","https"): return url
    return urljoin(base, url)

def best_from_srcset(srcset: str, base: str = PC_BASE) -> str:
    if not srcset: return ""
    best, sz = "", -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits: continue
        url = absolutize_url(bits[0], base)
        n = 0
        if len(bits) > 1:
            m = re.search(r"(\d+)(w|x)$", bits[1])
            if m: n = int(m.group(1))
        if n > sz: best, sz = url, n
    return best

def _clean_set_name(raw: str) -> str:
    s = (raw or "").strip()
    if s.lower().startswith("pokemon "): s = s[8:]
    low = s.lower()
    for pref in LANG_PREFIXES:
        if low.startswith(pref + " "):
            s = s[len(pref)+1:]
            break
    s = re.sub(r"\s*(Prices|Price Guide).*?$", "", s, flags=re.I).strip()
    return s

def _text_to_float_money(txt: str) -> float:
    if not txt: return 0.0
    s = txt.strip()
    # remove non-numeric except . and ,
    s = re.sub(r"[^\d\.,]", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0

def _find_price_in_soup(soup: BeautifulSoup, selectors: list[str]) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t: return t
    return ""

# -------- Unified fetch: name + set + raw price + psa10 + image (1-2 requests) ----------
def fetch_pc_details(lookup_id: str, usd_to_gbp: float) -> dict:
    """
    Returns dict: {name, set_name, price_gbp, psa10_gbp, image_url}
    price_gbp = loose/used price (GBP), psa10_gbp = graded 10 price (GBP, 0 if unavailable)
    """
    out = {"name":"", "set_name":"", "price_gbp":0.0, "psa10_gbp":0.0, "image_url":""}
    url = f"{PC_BASE}/game/{lookup_id}"

    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        if r.status_code != 200:
            return out
        soup = BeautifulSoup(r.text, "lxml")

        # ---- NAME + SET ----
        h1 = soup.find("h1", id="product_name") or soup.find("h1", class_="chart_title") or soup.find("h1")
        h1_text = " ".join(h1.get_text(" ", strip=True).split()) if h1 else ""
        og_title = (soup.find("meta", {"property":"og:title"}) or {}).get("content") or ""
        page_title = (soup.title.string if soup.title else "") or ""
        name_source = next((t for t in [h1_text, og_title, page_title] if t), "")
        if name_source:
            m = re.search(r"^(.*?)\s*#\d+", name_source)
            out["name"] = (m.group(1) if m else name_source).strip()

        a = None
        if h1: a = h1.find("a", href=re.compile(r"^/console/"))
        if not a: a = soup.find("a", href=re.compile(r"^/console/"))
        if a:
            out["set_name"] = _clean_set_name(a.get_text(" ", strip=True))
        else:
            tail_src = next((t for t in [h1_text, og_title, page_title] if t), "")
            if tail_src:
                parts = re.split(r"#\d+\s*", tail_src, maxsplit=1)
                if len(parts) > 1:
                    out["set_name"] = _clean_set_name(parts[1])

        # ---- RAW (used/loose) PRICE via embedded JSON (USD -> GBP) ----
        cents = None
        s1 = soup.find("script", string=re.compile(r"VGPC\.product\s*="))
        if s1:
            m = re.search(r"VGPC\.product\s*=\s*(\{.*?\});", s1.get_text(), re.S)
            if m:
                try:
                    prod = json.loads(m.group(1))
                    cents = prod.get("used-price") or prod.get("loose-price")
                except json.JSONDecodeError:
                    pass
        if cents is None:
            s2 = soup.find("script", string=re.compile(r"VGPC\.chart_data"))
            if s2:
                m2 = re.search(r"VGPC\.chart_data\s*=\s*(\{.*?\});", s2.get_text(), re.S)
                if m2:
                    try:
                        chart = json.loads(m2.group(1))
                        used = chart.get("used") or []
                        if used: cents = used[-1][1]
                    except json.JSONDecodeError:
                        pass
        if isinstance(cents,(int,float)):
            out["price_gbp"] = round((cents/100.0) * float(usd_to_gbp), 2)

        # ---- IMAGE: og:image -> itemprop=image ----
        og = soup.find("meta", attrs={"property":"og:image"})
        if og and og.get("content"): out["image_url"] = absolutize_url(og["content"])
        if not out["image_url"]:
            img = soup.find("img", attrs={"itemprop":"image"})
            if img and img.get("srcset"):
                out["image_url"] = best_from_srcset(img.get("srcset"))
            elif img and img.get("src"):
                out["image_url"] = absolutize_url(img.get("src"))

        # ---- PSA 10 PRICE: request grade=10 and scrape graded price cell ----
        try:
            r10 = requests.get(url + "?grade=10", headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
            if r10.status_code == 200:
                s10 = BeautifulSoup(r10.text, "lxml")
                # ONLY read the PSA-10 price from the exact element: #manual_only_price > span.price.js-price
                t10 = _find_price_in_soup(
                    s10,
                    [
                        "#manual_only_price span.price.js-price",
                        "td#manual_only_price span.price.js-price",
                    ],
                )
                if t10:
                    raw = (t10 or "").strip()
                    # If it's '-' or parses to 0, treat as unavailable -> 0
                    if raw in {"", "-", "0", "0.00", "£0.00", "$0.00"}:
                        out["psa10_gbp"] = 0.0
                    else:
                        val = _text_to_float_money(raw)
                        if val <= 0:
                            out["psa10_gbp"] = 0.0
                        else:
                            # If page shows GBP (has £) don't convert; if $, convert to GBP
                            if "£" in raw:
                                out["psa10_gbp"] = round(val, 2)
                            else:
                                out["psa10_gbp"] = round(val * float(usd_to_gbp), 2)
        except Exception:
            pass
            pass

            pass

    except Exception:
        return out

    return out

def cache_image_from_url(lookup_id: str, img_url: str) -> bool:
    if not img_url: return False
    try:
        r = requests.get(img_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
        if r.status_code != 200 or not r.content: return False
        with Image.open(io.BytesIO(r.content)) as im:
            if im.mode not in ("RGB","RGBA"): im = im.convert("RGBA")
            im.thumbnail((220,160))
            im.save(cached_image_path(lookup_id), format="WEBP", quality=85, method=6)
        return True
    except Exception:
        return False

def ensure_images_for_rows(rows: pd.DataFrame) -> int:
    downloaded = 0
    total = len(rows)
    if total == 0: return 0
    progress = st.progress(0)
    for i,(idx,row) in enumerate(rows.iterrows(), start=1):
        lookup_id = str(row["lookupID"]).strip()
        if lookup_id and not is_valid_image(cached_image_path(lookup_id)):
            details = fetch_pc_details(lookup_id, 1.0)  # only need image url
            if cache_image_from_url(lookup_id, details.get("image_url","")): downloaded += 1
        progress.progress(int(i/total*100))
    return downloaded

# --------- Merge helper: MERGE ONLY BY lookupID ----------
def merge_by_lookupid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Raw Price"] = pd.to_numeric(df["Raw Price"], errors="coerce").fillna(0.0)
    if "PSA 10 Price" not in df.columns:
        df["PSA 10 Price"] = 0.0
    df["PSA 10 Price"] = pd.to_numeric(df["PSA 10 Price"], errors="coerce").fillna(0.0)

    df_sorted = df.sort_values(["ID"], kind="stable")

    def first_nonempty(series):
        for v in series:
            if pd.notna(v) and str(v).strip():
                return v
        return ""

    def first_nonzero(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        nz = s[s > 0]
        return float(nz.iloc[0]) if len(nz) else 0.0

    merged = (df_sorted
              .groupby("lookupID", as_index=False)
              .agg(ID=("ID","min"),
                   Name=("Name", first_nonempty),
                   Set=("Set", first_nonempty),
                   Type=("Type", first_nonempty),
                   Quantity=("Quantity","sum"),
                   Raw_Price=("Raw Price", first_nonzero),
                   PSA10=("PSA 10 Price", first_nonzero))
             )

    merged["Raw Total"] = (merged["Quantity"] * merged["Raw_Price"]).round(2)
    merged = merged.rename(columns={"Raw_Price":"Raw Price", "PSA10":"PSA 10 Price"})
    return merged[["ID","lookupID","Name","Set","Type","Quantity","Raw Price","PSA 10 Price","Raw Total"]]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Pokémon Manager", layout="wide")
st.title("Pokémon Manager")

# --- Simple navigation (modern, pill-style) ---
# Use query params for navigation and custom-styled links.
_q = _get_query_params()

_active = _q.get("page", "Home")
if isinstance(_active, list):
    _active = _active[0]
if _active not in {"Home","Collections","Stock"}:
    _active = "Home"

# Sidebar look & feel + nav styles
# Title + navigation
st.sidebar.markdown('### Pokémon Manager')

# Radio navigation (no new tab)
opts = ['Home', 'Collections', 'Stock']
_q = _get_query_params()
_active = _q.get('page', 'Home')
if isinstance(_active, list):
    _active = _active[0]
if _active not in opts:
    _active = 'Home'
_choice = st.sidebar.radio('Navigation', opts, index=opts.index(_active), label_visibility='collapsed')
if _choice != _active:
    try:
        st.query_params['page'] = _choice
        st.rerun()
    except Exception:
        pass
_page_choice = _choice

# Divider before options
st.sidebar.markdown("---")
st.sidebar.markdown('**Options**')


with st.sidebar.expander("Price Fetching", expanded=False):
    usd_to_gbp = st.number_input("USD → GBP rate", min_value=0.01, max_value=10.0, value=float(st.session_state.saved_rate), step=0.01)
    st.session_state.saved_rate = float(usd_to_gbp)
    force_update = st.checkbox("Force update all prices", value=False)
    auto_merge = st.checkbox("Auto-merge after actions", value=True)
    update_only_current_set = st.checkbox("Force update shown cards (respects Set/Search filters)", value=False)
    update_btn = st.button("🔁 Update Prices", use_container_width=True)
    save_rate_btn = st.button("💾 Save rate", use_container_width=True)
    merge_btn = st.button("🧹 Merge duplicates (now)", use_container_width=True)

if save_rate_btn:
    save_rate(usd_to_gbp); st.session_state.saved_rate = usd_to_gbp; st.sidebar.success("Rate saved")

with st.sidebar.expander("Image Fetching", expanded=False):
    images_only_current_set = st.checkbox("Limit to current set", value=False)
    fetch_missing_btn  = st.button("🖼️ Fetch images (missing only)", use_container_width=True)
    refresh_broken_btn = st.button("🔁 Refresh images (missing or broken)", use_container_width=True)

# Load CSV

try:
    df = pd.read_csv(CSV_PATH, delimiter=";")
except FileNotFoundError:
    st.error(f"CSV not found: {CSV_PATH}"); st.stop()

# Ensure new column exists
if "PSA 10 Price" not in df.columns:
    df["PSA 10 Price"] = 0.0

needed = {"ID","lookupID","Name","Set","Type","Quantity","Raw Price","PSA 10 Price"}
miss = needed - set(df.columns)
if miss:
    st.error("CSV missing: " + ", ".join(sorted(miss))); st.stop()

df["Raw Price"]   = pd.to_numeric(df["Raw Price"], errors="coerce").fillna(0.0).round(2)
df["PSA 10 Price"]= pd.to_numeric(df["PSA 10 Price"], errors="coerce").fillna(0.0).round(2)
df["Quantity"]    = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
df["Raw Total"]   = (df["Quantity"] * df["Raw Price"]).round(2)




# Summary




# --- Stock page ---
if _page_choice == "Stock":
    st.markdown("### 📦 Stock")
    STOCK_CSV = "stock.csv"

    # Load or initialize stock data
    try:
        stock_df = pd.read_csv(STOCK_CSV, sep=";")
    except FileNotFoundError:
        stock_df = pd.DataFrame([{"RowID": 0, "Product": "", "Amount": 0}])
        stock_df.to_csv(STOCK_CSV, sep=";", index=False)
    except Exception as e:
        st.error(f"Failed to read {STOCK_CSV}: {e}")
        stock_df = pd.DataFrame([{"RowID": 0, "Product": "", "Amount": 0}])

    # Ensure needed columns
    if "Product" not in stock_df.columns: stock_df["Product"] = ""
    if "Amount" not in stock_df.columns: stock_df["Amount"] = 0
    if "RowID" not in stock_df.columns:
        stock_df.insert(0, "RowID", range(len(stock_df)))

    # Coerce types
    stock_df["Product"] = stock_df["Product"].astype(str)
    stock_df["Amount"] = pd.to_numeric(stock_df["Amount"], errors="coerce").fillna(0).astype(int)
    stock_df["RowID"]  = pd.to_numeric(stock_df["RowID"], errors="coerce").fillna(0).astype(int)

    st.caption("Add, edit, or remove items below. Click **Save** to persist changes.")
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, JsCode

    # Build editable grid
    gb2 = GridOptionsBuilder.from_dataframe(stock_df[["RowID","Product","Amount"]], enableRowGroup=False, enableValue=False, enablePivot=False)
    gb2.configure_default_column(editable=True, resizable=True, minWidth=140)
    gb2.configure_column("RowID", hide=True)
    gb2.configure_column("Product", header_name="Product", editable=True)
    gb2.configure_column("Amount", header_name="Amount", type=["numericColumn"], editable=True, valueParser=JsCode("function(p){var v=Number(p.newValue);return isFinite(v)?Math.max(0,Math.round(v)):p.oldValue;}"))
    gb2.configure_selection("multiple", use_checkbox=True)
    gb2.configure_grid_options(getRowId=JsCode("function(p){return String(p.data && p.data.RowID!=null ? p.data.RowID : p.rowIndex);}"),
                               domLayout="normal",
                               rowHeight=44,
                               headerHeight=40,
                               singleClickEdit=True)

    grid2 = AgGrid(
        stock_df,
        gridOptions=gb2.build(),
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme="balham",
        height=480
    )

    colA, colB, colC = st.columns(3)
    save_stock = colA.button("💾 Save", use_container_width=True)
    add_row    = colB.button("➕ Add row", use_container_width=True)
    del_rows   = colC.button("🗑️ Delete selected", use_container_width=True)

    # Helper to persist and rerun
    def _persist_and_rerun(df):
        # Re-index RowID for stability
        df = df.copy().reset_index(drop=True)
        if "RowID" in df.columns:
            df["RowID"] = range(len(df))
            # Ensure desired column order
            cols = ["RowID"] + [c for c in df.columns if c != "RowID"]
            df = df[cols]
        else:
            df.insert(0, "RowID", range(len(df)))
        df.to_csv(STOCK_CSV, sep=";", index=False)
        st.success("Saved.")
        st.rerun()

    # Save edits
    if save_stock:
        try:
            edited = None
            if isinstance(grid2, dict) and "data" in grid2:
                edited = pd.DataFrame(grid2["data"])
            elif hasattr(grid2, "data"):
                edited = pd.DataFrame(grid2.data)
            if edited is None or edited.empty:
                edited = stock_df.copy()

            # Keep only required columns
            edited = edited[["RowID","Product","Amount"]].copy()
            edited["Product"] = edited["Product"].astype(str).str.strip()
            edited["Amount"] = pd.to_numeric(edited["Amount"], errors="coerce").fillna(0).astype(int)

            _persist_and_rerun(edited)
        except Exception as e:
            st.error(f"Failed to save: {e}")

    # Add a blank row
    if add_row:
        new = pd.DataFrame([{"Product": "", "Amount": 0}])
        _persist_and_rerun(pd.concat([stock_df, new], ignore_index=True))

    # Delete selected rows
    if del_rows:
        try:
            sel = []
            if isinstance(grid2, dict) and "selected_rows" in grid2:
                sel = grid2["selected_rows"]
            elif hasattr(grid2, "selected_rows"):
                sel = grid2.selected_rows
            sel_ids = set()
            for r in sel or []:
                # selection rows can be dicts
                if isinstance(r, dict) and "RowID" in r:
                    sel_ids.add(int(r["RowID"]))
            if not sel_ids:
                st.warning("No rows selected.")
            else:
                keep = stock_df[~stock_df["RowID"].isin(list(sel_ids))]
                _persist_and_rerun(keep)
        except Exception as e:
            st.error(f"Failed to delete: {e}")

    st.stop()

# --- Collections page ---
if _page_choice == "Collections":
    st.markdown("### 🗂️ Collections")
    # Files live in ./collections/
    COLLECTIONS_DIR = "collections"
    COLLECTIONS_CONFIG_CSV = os.path.join(COLLECTIONS_DIR, "config.csv")
    COLLECTIONS_CACHE = os.path.join(COLLECTIONS_DIR, "cache")

    # Load collections config
    coll_cfg = None
    try:
        coll_cfg = pd.read_csv(COLLECTIONS_CONFIG_CSV, sep=";")
    except FileNotFoundError:
        st.error(f"Couldn't find collections config CSV at: {COLLECTIONS_CONFIG_CSV}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read {COLLECTIONS_CONFIG_CSV}: {e}")
        st.stop()

    # Expect columns: Name, fileName (plus others like id/link ignored)
    required_cols = {"Name", "fileName"}
    missing = required_cols - set(map(str, coll_cfg.columns))
    if missing:
        st.error("Config CSV missing required columns: " + ", ".join(sorted(missing)))
        st.stop()

    names = coll_cfg["Name"].astype(str).tolist()
    selected = st.selectbox("Choose a collection", names, index=0, key="collections_select")

    row = coll_cfg[coll_cfg["Name"].astype(str) == selected].iloc[0]
    set_file = os.path.join(COLLECTIONS_DIR, str(row["fileName"]).strip())

    # Load the chosen set CSV (comma-delimited)
    try:
        set_df = pd.read_csv(set_file)
    except FileNotFoundError:
        st.error(f"Set CSV not found: {set_file}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to read set CSV '{set_file}': {e}")
        st.stop()

    if "lookupid" not in set_df.columns:
        st.error("Set CSV must contain a 'lookupid' column (e.g. 'pokemon-japanese-mega-brave/bulbasaur-1').")
        st.stop()
    # Map of lookupid -> set number (lowercased key) for tooltips
    set_number_map = {}
    try:
        if "set number" in set_df.columns:
            tmp_num = set_df[["lookupid", "set number"]].dropna(subset=["lookupid"]).copy()
            tmp_num["lk_lower"] = tmp_num["lookupid"].astype(str).str.strip().str.lower()
            set_number_map = dict(zip(tmp_num["lk_lower"], tmp_num["set number"]))
    except Exception:
        set_number_map = {}

    # Load ownership list from db.csv (by lookupid); try semicolon first, then comma
    owned_lookup = set()
    try:
        _db = None
        for _sep in (";", ","):
            try:
                _db = pd.read_csv("db.csv", sep=_sep)
                break
            except Exception:
                _db = None
        if _db is not None:
            cols_norm = {str(c).strip().lower(): c for c in _db.columns}
            lk_col = cols_norm.get("lookupid") or cols_norm.get("lookupid".lower()) or cols_norm.get("lookup_id") or cols_norm.get("lookupid ")
            if lk_col is None and "lookupID" in _db.columns:
                lk_col = "lookupID"
            if lk_col is None and "lookupid" in _db.columns:
                lk_col = "lookupid"
            if lk_col is None:
                # Try from main app dataframe 'df' if present
                try:
                    if "lookupID" in df.columns:
                        _db = df[["lookupID"]].copy()
                        lk_col = "lookupID"
                except Exception:
                    pass
            if lk_col is not None:
                owned_lookup = {str(v).strip().lower() for v in _db[lk_col].dropna().astype(str)}
    except Exception:
        owned_lookup = set()

    
    # Sort strictly by 'set number' if available
    if "set number" in set_df.columns:
        try:
            set_df = set_df.copy()
            set_df["set number"] = pd.to_numeric(set_df["set number"], errors="coerce")
            set_df = set_df.sort_values("set number", na_position="last")
        except Exception:
            set_df = set_df.sort_values("lookupid")
    else:
        set_df = set_df.sort_values("lookupid")



    # Build gallery records
    records = []
    # Optional: map lookupID -> Name from the main db, if available
    name_map = {}
    try:
        if "lookupID" in df.columns and "Name" in df.columns:
            # Prefer non-empty names
            tmp = df.loc[df["lookupID"].notna(), ["lookupID", "Name"]].copy()
            tmp["Name"] = tmp["Name"].fillna("")
            for k, v in zip(tmp["lookupID"].astype(str), tmp["Name"].astype(str)):
                if k not in name_map or (not name_map[k] and v):
                    name_map[k] = v
    except Exception:
        pass

    for lk in set_df["lookupid"].astype(str):
        if "/" in lk:
            set_lk, card_lk = lk.split("/", 1)
        else:
            # Fallback if malformed; treat entire as card id
            set_lk, card_lk = "", lk
        img_path = os.path.join(COLLECTIONS_CACHE, set_lk, f"{card_lk}.jpg")
        # Prefer friendly name if we know it, else derive from last segment
        nice_name = name_map.get(lk, card_lk.replace("-", " ").title())
        nice_name = urllib.parse.unquote(nice_name)
        records.append({"lookupid": lk, "name": nice_name, "image": img_path, "exists": os.path.exists(img_path), "owned": (str(lk).strip().lower() in owned_lookup), "set_number": set_number_map.get(str(lk).strip().lower())})

    # Summary line
    total = len(records)
    have = sum(1 for r in records if r["exists"])
    missing = total - have
    owned = sum(1 for r in records if r['owned'])
    # Render a tighter, centered gallery using HTML/CSS + base64 images
    def _img_to_data_uri(p):
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return None

    # Ownership summary with percent
    owned = sum(1 for r in records if r.get('owned'))
    own_pct = int(round((owned / max(total, 1)) * 100))
    st.caption(f"{have}/{total} images found • {missing} missing • {owned}/{total} owned • {own_pct}%")
    st.progress(own_pct)
    st.markdown("""
    <style>
      .pm-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 8px; transition: transform .12s ease, box-shadow .12s ease, border .12s ease; }
      .pm-card:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 6px 16px rgba(0,0,0,.12); border-color: #ffffff; }
      .pm-num { font-size: 0.75em; color: #6b7280; display: inline-block; margin-top: 2px; }
    </style>
    """, unsafe_allow_html=True)


    cards_html = ["<div style='display:flex;flex-wrap:wrap;gap:14px;justify-content:center;'>"]
    # Image sizing constants
    IMG_W = 180  # image width in pixels (crisper, larger)
    PLACEHOLDER_H = int(IMG_W * 1.5)  # card aspect approx 2:3

    for rec in records:
        # Tooltip text for hover (compute inside loop)
        sn = rec.get("set_number")
        try:
            import math
            is_nan = isinstance(sn, float) and math.isnan(sn)
        except Exception:
            is_nan = False
        tip_num = f" #{sn}" if (sn not in (None, "", "nan") and not is_nan) else ""
        tip_owned = "Owned" if rec.get("owned", False) else "Not owned"
        # Fix possessive capitalization and decode
        name_fixed = urllib.parse.unquote(rec['name'])
        name_fixed = re.sub(r"\'S", "'s", name_fixed)
        tooltip = f"{name_fixed}{tip_num} · {tip_owned}"

        # Build image (owned -> color, not owned -> grayscale + red border + opacity + ❌ overlay)
        if rec["exists"]:
            data_uri = _img_to_data_uri(rec["image"])
            if data_uri:
                owned_flag = rec.get("owned", False)
                base_styles = f"width:{IMG_W}px;object-fit:contain;display:block;margin:0 auto;border-radius:8px;"
                extra = "filter:grayscale(100%);opacity:0.75;border:2px solid #ef4444;" if not owned_flag else ""
                img_tag = f"<img src='{data_uri}' style='{base_styles}{extra}'>"
                overlay = ("<div style='position:absolute;top:4px;right:6px;font-size:22px;opacity:0.55;"
                           "user-select:none;pointer-events:none;'>❌</div>") if not owned_flag else ""
                pc_url = f"https://www.pricecharting.com/game/{urllib.parse.quote(rec['lookupid'], safe='/')}"
                img_html = ("<div style='position:relative;display:inline-block;'>"
                            f"<a href=\"{pc_url}\" target=\"_blank\" rel=\"noopener\" title=\"{tooltip}\">"
                            f"{img_tag}</a>{overlay}</div>")
            else:
                pc_url = f"https://www.pricecharting.com/game/{urllib.parse.quote(rec['lookupid'], safe='/')}"
                placeholder_div = (f"<div style='width:{IMG_W}px;height:{PLACEHOLDER_H}px;border:1px dashed #94a3b8;border-radius:8px;"
                                   f"display:flex;align-items:center;justify-content:center;color:#64748b;margin:0 auto;'>Missing</div>")
                img_html = ("<div style='position:relative;display:inline-block;'>"
                            f"<a href=\"{pc_url}\" target=\"_blank\" rel=\"noopener\" title=\"{tooltip}\">"
                            f"{placeholder_div}</a></div>")
        else:
            pc_url = f"https://www.pricecharting.com/game/{urllib.parse.quote(rec['lookupid'], safe='/')}"
            placeholder_div = (f"<div style='width:{IMG_W}px;height:{PLACEHOLDER_H}px;border:1px dashed #94a3b8;border-radius:8px;"
                               f"display:flex;align-items:center;justify-content:center;color:#64748b;margin:0 auto;'>Missing</div>")
            img_html = ("<div style='position:relative;display:inline-block;'>"
                        f"<a href=\"{pc_url}\" target=\"_blank\" rel=\"noopener\" title=\"{tooltip}\">"
                        f"{placeholder_div}</a></div>")
# Build caption with clean name and separate set number
        # - Decode & fix possessives
        name_fixed = urllib.parse.unquote(rec['name'])
        name_fixed = re.sub(r"'S\b", "'s", name_fixed)
        # - Remove trailing numbers in name
        name_fixed = re.sub(r"\s+\d+$", "", name_fixed)

        num = rec.get("set_number")
        try:
            import math
            is_nan = isinstance(num, float) and math.isnan(num)
        except Exception:
            is_nan = False

        if (num not in (None, "", "nan") and not is_nan):
            try:
                num_str = str(int(num))
            except Exception:
                num_str = str(num)
            caption_text = f"{name_fixed}<br>{num_str}"
        else:
            caption_text = name_fixed

        caption_html = f"<div style='text-align:center;font-size:0.85em;color:#374151;margin-top:4px;width:{IMG_W}px;'> {caption_text}</div>"
        cards_html.append(f"<div class='pm-card' style='flex:0 0 auto;text-align:center;'>{img_html}{caption_html}</div>")

    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)
    st.stop()




st.markdown("### 📊 Collection Summary")

st.markdown("""
<style>
/* Rounded card background behind Streamlit metrics */
div[data-testid="stMetric"] {
  background: #ffffff;
  border-radius: 16px;
  padding: 16px 18px;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
/* Keep titles/values readable; don't alter sizes */
div[data-testid="stMetric"] label, 
div[data-testid="stMetricValue"] {
  color: inherit;
}
</style>
""", unsafe_allow_html=True)

# Ensure column exists
if "PSA 10 Price" not in df.columns:
    df["PSA 10 Price"] = 0.0

# Totals
q = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
raw = pd.to_numeric(df["Raw Price"], errors="coerce").fillna(0.0)
psa10 = pd.to_numeric(df["PSA 10 Price"], errors="coerce").fillna(0.0)

total_cards   = int(q.sum())
unique_cards  = int(df["lookupID"].astype(str).nunique())
total_raw     = float((q * raw).sum())
total_psa10   = float((q * psa10).sum())

# Metrics row (adds PSA-10 next to Raw)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cards", f"{total_cards:,}")
c2.metric("Total Unique Cards", f"{unique_cards:,}")
c3.metric("Total Raw Value (GBP)", f"£{total_raw:,.2f}")
c4.metric("Total PSA 10 Value (GBP)", f"£{total_psa10:,.2f}")


# Breakdown
set_breakdown = (
    df.groupby("Set", dropna=False, as_index=False)
      .agg(Total_Cards=("Quantity","sum"), Total_Value_GBP=("Raw Total","sum"))
      .sort_values("Total_Value_GBP", ascending=False)
)
set_breakdown["Total_Value_GBP"] = set_breakdown["Total_Value_GBP"].round(2)

def set_button_html(s):
    if pd.isna(s) or not str(s).strip(): return ""
    url = f"{PC_BASE}/search-products?type=prices&q={quote_plus(str(s))}"
    return f'<a href="{url}" target="_blank"><button class="pm-open-btn">Open</button></a>'

st.markdown("""
<style>
.pm-table{width:100%;border-collapse:separate;border-spacing:0;}
.pm-table th,.pm-table td{padding:10px 12px;border-bottom:1px solid rgba(0,0,0,.06);}
.pm-table th{text-align:left;font-weight:600;color:#334155;background:#fff;}
.pm-open-btn{padding:8px 14px;border:none;border-radius:9999px;background:linear-gradient(180deg,#10b981,#0ea672);color:#fff;font-weight:700;cursor:pointer;box-shadow:0 2px 6px rgba(16,185,129,.25);}
</style>
""", unsafe_allow_html=True)

with st.expander("📚 Breakdown by Set", expanded=False):
    show_df = set_breakdown.rename(columns=lambda c: c.replace("_"," "))
    show_df["PriceCharting"] = show_df["Set"].apply(set_button_html)
    st.markdown("""
<style>
.pm-set-scroll{max-height:480px;overflow-y:auto;border:1px solid rgba(0,0,0,.06);border-radius:12px;}
.pm-set-scroll .pm-table{margin:0;}
.pm-set-scroll .pm-table th{position:sticky;top:0;background:#fff;z-index:2;}
</style>
""", unsafe_allow_html=True)
    st.markdown(f'<div class="pm-set-scroll">{show_df.to_html(escape=False, index=False, classes="pm-table")}</div>', unsafe_allow_html=True)
    try:
        st.bar_chart(set_breakdown.set_index("Set")["Total_Value_GBP"])
    except Exception:
        pass

# ---------------- Add New Card (collapsible, default closed) ----------------
with st.expander("➕ Add New Card", expanded=False):
    with st.form("add_card_form", clear_on_submit=True):
        pc_url = st.text_input(
            "PriceCharting URL (or lookupID)",
            placeholder="https://www.pricecharting.com/game/pokemon-japanese-mega-brave/vulpix-67"
        )
        qty = st.number_input("Quantity", min_value=0, value=1, step=1)
        type_choice = st.selectbox("Type", ["Normal", "Holo", "Reverse Holo"], index=0)

        colA, colB = st.columns(2)
        fetch_now = colA.checkbox("Fetch price now (GBP)", value=True)
        fetch_img = colB.checkbox("Fetch image now", value=True)

        manual_price = st.number_input("Manual Raw Price (GBP) (optional)", min_value=0.0, value=0.0, step=0.01)
        add_clicked = st.form_submit_button("Add Card")

if 'add_clicked' in locals() and add_clicked:
    lookup_id = parse_lookup_id(pc_url)
    if not lookup_id:
        st.error("Please paste a valid PriceCharting URL (or lookupID).")
    else:
        details = fetch_pc_details(lookup_id, usd_to_gbp)
        name_val = (details.get("name") or "").strip()
        set_val  = (details.get("set_name") or "").strip()

        # Raw price: manual > fetched > 0
        price_gbp = 0.0
        if manual_price and float(manual_price) > 0:
            price_gbp = round(float(manual_price), 2)
        elif fetch_now and details.get("price_gbp", 0) > 0:
            price_gbp = float(details["price_gbp"])

        psa10_gbp = float(details.get("psa10_gbp", 0.0))

        try:
            next_id = int(pd.to_numeric(df["ID"], errors="coerce").fillna(0).max()) + 1 if len(df) else 1
        except Exception:
            next_id = 1

        new = {
            "ID": int(next_id),
            "lookupID": lookup_id,
            "Name": name_val,
            "Set": set_val,
            "Type": type_choice,
            "Quantity": int(qty),
            "Raw Price": round(price_gbp, 2),
            "PSA 10 Price": round(psa10_gbp, 2),
        }
        new["Raw Total"] = round(new["Quantity"] * new["Raw Price"], 2)

        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)

        if fetch_img:
            cache_image_from_url(lookup_id, details.get("image_url",""))

        df.to_csv(CSV_PATH, sep=";", index=False)
        st.success(f"Added {name_val or '(no name)'} in set {set_val or '(unknown)'} — ID {next_id}")
        st.rerun()

# ---------------- Card List (filters above table) ----------------

# ---------------- Bulk Add Cards (paste multiple links or lookupIDs) ----------------
with st.expander("➕ Bulk Add Cards (multiple)", expanded=False):
    st.caption("Paste one PriceCharting URL or lookupID per line. Empty lines are ignored. Duplicates are de-duplicated.")
    bulk_text = st.text_area(
        "Links or lookupIDs (one per line)",
        height=160,
        placeholder=("https://www.pricecharting.com/game/pokemon-japanese-mega-brave/vulpix-67\n"
                    "pokemon-japanese-mega-brave/pikachu-25\n"
                    "https://www.pricecharting.com/game/pokemon-english-scarlet-violet/charizard-201")
    )
    colQ, colT = st.columns([1,1])
    bulk_qty  = colQ.number_input("Default Quantity per card", min_value=0, value=1, step=1)
    bulk_type = colT.selectbox("Type (applies to all)", ["Normal", "Holo", "Reverse Holo"], index=0)
    colA, colB, colC = st.columns(3)
    bulk_fetch_price = colA.checkbox("Fetch price now (GBP)", value=True)
    bulk_fetch_img   = colB.checkbox("Fetch image now", value=True)
    # Optional: respect saved rate (already in usd_to_gbp)
    st.caption(f"Using USD→GBP rate: {usd_to_gbp}")
    run_bulk = st.button("Add All", use_container_width=True)

if "run_bulk" in locals() and run_bulk:
    # Parse lines -> lookupIDs
    lines = [ln.strip() for ln in (bulk_text or "").splitlines()]
    lookups = []
    for ln in lines:
        if not ln:
            continue
        try:
            lk = parse_lookup_id(ln)
            if lk:
                lookups.append(lk)
        except Exception:
            pass
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for lk in lookups:
        if lk not in seen:
            uniq.append(lk); seen.add(lk)

    if not uniq:
        st.warning("Nothing to add. Paste at least one valid URL or lookupID.")
    else:
        to_add = []
        errs = []
        added = 0
        st.info(f"Processing {len(uniq)} item(s)…")
        prog_text = st.empty()
        progress = st.progress(0)
        try:
            # Compute next starting ID once
            try:
                next_id = int(pd.to_numeric(df["ID"], errors="coerce").fillna(0).max()) + 1 if len(df) else 1
            except Exception:
                next_id = 1

            for i, lk in enumerate(uniq, start=1):
                label = f"Fetching {lk} ({i}/{len(uniq)})"
                try:
                    progress.progress(int(i/max(len(uniq),1)*100), text=label)
                except TypeError:
                    prog_text.write(label)
                    progress.progress(int(i/max(len(uniq),1)*100))

                # Fetch details
                details = {}
                try:
                    details = fetch_pc_details(lk, usd_to_gbp if bulk_fetch_price else 1.0)
                except Exception as e:
                    errs.append((lk, str(e)))
                    details = {}

                name_val = (details.get("name") or "").strip()
                set_val  = (details.get("set_name") or "").strip()

                price_gbp = 0.0
                if bulk_fetch_price and details.get("price_gbp", 0) > 0:
                    price_gbp = float(details["price_gbp"])

                psa10_gbp = float(details.get("psa10_gbp", 0.0)) if bulk_fetch_price else 0.0

                new_row = {
                    "ID": int(next_id),
                    "lookupID": lk,
                    "Name": name_val,
                    "Set": set_val,
                    "Type": bulk_type,
                    "Quantity": int(bulk_qty),
                    "Raw Price": round(price_gbp, 2),
                    "PSA 10 Price": round(psa10_gbp, 2),
                }
                new_row["Raw Total"] = round(new_row["Quantity"] * new_row["Raw Price"], 2)
                to_add.append(new_row)
                added += 1
                next_id += 1

                if bulk_fetch_img:
                    try:
                        cache_image_from_url(lk, details.get("image_url",""))
                    except Exception:
                        pass

            # Append and save
            if to_add:
                df = pd.concat([df, pd.DataFrame(to_add)], ignore_index=True)
                df.to_csv(CSV_PATH, sep=";", index=False)
                st.success(f"Added {added} card(s) from bulk paste.")
                if errs:
                    with st.expander("Show fetch warnings", expanded=False):
                        for lk, msg in errs:
                            st.write(f"- {lk}: {msg}")
                st.rerun()
            else:
                st.warning("No rows were added.")
        except Exception as e:
            st.error(f"Bulk add failed: {e}")
            # No rerun on error


st.markdown("### 📋 Card List")
all_sets = sorted(df["Set"].dropna().astype(str).unique().tolist())
fc1, fc2 = st.columns(2)
selected_set = fc1.selectbox("Filter by Set (table only)", ["All sets"] + all_sets, index=0)
search_query  = fc2.text_input("Search Pokémon (table only)", placeholder="Type a name…").strip()

table_df = df.copy()
if selected_set != "All sets":
    table_df = table_df[table_df["Set"].astype(str) == selected_set]
if search_query:
    pat = re.escape(search_query)
    table_df = table_df[table_df["Name"].astype(str).str.contains(pat, case=False, na=False)]

# ---- Build grid ----
grid_df = table_df.copy().reset_index(drop=True)
grid_df["Raw Price"]    = pd.to_numeric(grid_df["Raw Price"], errors="coerce").round(2)
grid_df["PSA 10 Price"] = pd.to_numeric(grid_df["PSA 10 Price"], errors="coerce").round(2)
grid_df["Raw Total"]    = pd.to_numeric(grid_df["Raw Total"], errors="coerce").round(2)
grid_df["Open"] = ""; grid_df["Adjust"] = ""; grid_df["Save"] = ""; grid_df["ID"] = grid_df["ID"].astype(str)

def image_uri_for_lookup(lookup_id: str) -> str:
    p = cached_image_path(str(lookup_id))
    return path_to_data_uri(p) if is_valid_image(p) else ""
grid_df["ImageURI"] = grid_df["lookupID"].apply(image_uri_for_lookup)

grid_df["OrigQuantity"] = grid_df["Quantity"].copy()
# BigInt-safe currency formatter
currency_fmt = JsCode("""
function(p){
  const v = p.value;
  const n = (typeof v === 'bigint') ? Number(v) : parseFloat(v);
  return (Number.isFinite(n)) ? '£' + n.toFixed(2) : '';
}
""")
# BigInt-safe raw total getter
total_getter  = JsCode("""
function(p){
  const qv = p.data ? p.data.Quantity : 0;
  const rv = p.data ? p.data['Raw Price'] : 0;
  const q  = (typeof qv === 'bigint') ? Number(qv) : parseFloat(qv);
  const r  = (typeof rv === 'bigint') ? Number(rv) : parseFloat(rv);
  return (Number.isFinite(q) ? q : 0) * (Number.isFinite(r) ? r : 0);
}
""")

open_renderer = JsCode(f"""
class OpenCellRenderer {{
  init(p){{
    const stop=e=>{{if(e){{e.preventDefault();e.stopPropagation();}}}};
    const b=document.createElement('button');
    b.innerText='Open';
    Object.assign(b.style,{{padding:'8px 14px',border:'none',borderRadius:'9999px',background:'linear-gradient(180deg,#10b981,#0ea672)',color:'#fff',fontWeight:'700',boxShadow:'0 2px 6px rgba(16,185,129,.25)',cursor:'pointer'}});
    b.addEventListener('mousedown',stop);
    b.addEventListener('click',e=>{{stop(e); if(p.data&&p.data.lookupID) window.open('{PC_BASE}/game/'+p.data.lookupID,'_blank'); p.api.clearFocusedCell();}});
    this.eGui=b;
  }}
  getGui(){{return this.eGui;}}
}}
""")

adjust_renderer = JsCode("""
class AdjustRenderer{
  init(p){
    const stop=e=>{if(e){e.preventDefault();e.stopPropagation();}};
    const W=document.createElement('div'); Object.assign(W.style,{display:'flex',gap:'8px',justifyContent:'flex-end',alignItems:'center',width:'100%'});
    const mk=(t,bg)=>{const b=document.createElement('button'); b.innerText=t; Object.assign(b.style,{width:'32px',height:'32px',border:'none',borderRadius:'9999px',background:bg,color:'#fff',fontWeight:'900',cursor:'pointer',boxShadow:'0 1px 3px rgba(0,0,0,.15)'}); b.addEventListener('mousedown',stop); return b;};
    const minus=mk('−','#ef4444'), plus=mk('+','#10b981');
    const upd=d=>{const cur=Number(p.node.data.Quantity)||0; let n=cur+d; if(n<0)n=0; p.node.setDataValue('Quantity',n); p.api.clearFocusedCell();};
    minus.addEventListener('click',e=>{stop(e);upd(-1);}); plus.addEventListener('click',e=>{stop(e);upd(+1);});
    W.append(minus,plus); this.eGui=W;
  }
  getGui(){return this.eGui;}
}
""")


# Small thumbnail + name renderer for Name column
name_thumb_renderer = JsCode("""
class NameThumbRenderer {
  init(p){
    const wrap = document.createElement('div');
    Object.assign(wrap.style, { display:'flex', alignItems:'center', gap:'10px' });

    const src = (p.data && p.data.ImageURI) ? p.data.ImageURI : '';
    if (src){
      const img = document.createElement('img');
      Object.assign(img.style, {
        width:'32px',
        height:'48px',
        objectFit:'contain',
        borderRadius:'6px',
        boxShadow:'0 1px 2px rgba(0,0,0,0.12)'
      });
      img.src = src;
      wrap.appendChild(img);
    }

    const text = document.createElement('span');
    text.textContent = (p.data && p.data.Name) ? p.data.Name : '';
    wrap.appendChild(text);

    this.eGui = wrap;
  }
  getGui(){ return this.eGui; }
}
""")

# Simple tooltip (ID + Name) for Name column
name_simple_tooltip = JsCode("""
function(p){
  var d = (p && p.data) ? p.data : null;
  if(!d) return '';
  var id = (d.ID != null) ? String(d.ID) : '';
  var nm = d.Name || '';
  return 'ID: ' + id + ' — ' + nm;
}
""")
# Tooltip JS (compact + 5s auto-hide)
changes_ph = st.empty()
gb = GridOptionsBuilder.from_dataframe(
    grid_df[["Open", "Name", "Set", "Type", "Quantity", "Raw Price", "PSA 10 Price", "Raw Total", "Adjust", "Save", "lookupID", "ID", "ImageURI", "OrigQuantity"]],
    enableRowGroup=False, enableValue=False, enablePivot=False
)
gb.configure_default_column(resizable=True, sortable=True, filter=True, minWidth=120, editable=False)

# Name column: small thumbnail + simple tooltip
gb.configure_column("Name", cellRenderer=name_thumb_renderer, tooltipValueGetter=name_simple_tooltip, minWidth=220)
gb.configure_column("Open", header_name="PriceCharting", cellRenderer=open_renderer, sortable=False, filter=False, width=120)
gb.configure_column("Adjust", header_name="Qty", cellRenderer=adjust_renderer, sortable=False, filter=False, width=140, pinned="right")

gb.configure_column(
    "Quantity",
    type=["numericColumn"],
    editable=True,
    valueParser=JsCode("function(p){var v=Number(p.newValue); return isFinite(v)?Math.round(v):p.oldValue;}"),
    cellClass="qty-edit",
    width=120
)
gb.configure_column("Raw Price", type=["rightAligned"], valueFormatter=currency_fmt)
gb.configure_column("PSA 10 Price", type=["rightAligned"], valueFormatter=currency_fmt)
gb.configure_column("Raw Total", type=["rightAligned"], valueGetter=total_getter, valueFormatter=currency_fmt)
gb.configure_column("lookupID", hide=True); gb.configure_column("ID", hide=True); gb.configure_column("ImageURI", hide=True); gb.configure_column("OrigQuantity", hide=True)
gb.configure_grid_options(
    getRowId=JsCode("function(p){return String(p.data&&p.data.ID!=null?p.data.ID:p.rowIndex);}"),
    domLayout="normal", rowHeight=60, headerHeight=46,
    singleClickEdit=True, suppressClickEdit=False, suppressCellFocus=False,
        onGridReady=JsCode("""
      function(p){
        const fit=()=>{try{p.api.sizeColumnsToFit();}catch(_){}}; fit(); setTimeout(fit,50);
        if(!window._pmResizeReg){window.addEventListener('resize',()=>setTimeout(fit,60)); window._pmResizeReg=true;}
        function hideTip(){try{if(window._pmTipTimer)clearTimeout(window._pmTipTimer);}catch(_){}
          var t=document.getElementById('pm-row-tooltip'); if(t){t.style.display='none';} window._pmTipVisible=false;}
        try{
          var iframeEl=window.frameElement;
          if(iframeEl&&!iframeEl._pmLeaveBound){
            iframeEl.addEventListener('mouseleave',hideTip);
            iframeEl.addEventListener('mouseout',function(ev){var to=ev.relatedTarget||ev.toElement; if(!to||to.ownerDocument!==document){hideTip();}});
            iframeEl._pmLeaveBound=true;
          }
        }catch(e){}
        document.addEventListener('mouseleave',hideTip,true); window.addEventListener('blur',hideTip);
        document.addEventListener('visibilitychange',function(){if(document.hidden)hideTip();});
        const roots=document.querySelectorAll('.ag-root-wrapper,.ag-theme-balham,.ag-root');
        roots.forEach(root=>{if(!root._pmMoveBound){
          root.addEventListener('mousemove',function(ev){
            const tip=document.getElementById('pm-row-tooltip'); if(!tip||tip.style.display==='none')return;
            const pad=16,vw=window.innerWidth,vh=window.innerHeight; const r=tip.getBoundingClientRect(),w=r.width||260,h=r.height||190;
            let x=ev.clientX+pad,y=ev.clientY+pad; if(x+w>vw)x=vw-w-pad; if(y+h>vh)y=vh-h-pad; tip.style.left=x+'px'; tip.style.top=y+'px';
          });
          root.addEventListener('mouseleave',hideTip); root._pmMoveBound=true; }});
      }
    """)
)
grid_options = gb.build()

custom_css = {
    ".ag-cell-inline-editing .ag-input-field-input, .ag-cell-inline-editing input": {
        "font-size":"18px", "height":"36px", "line-height":"36px",
        "padding":"0 8px", "width":"100%", "box-sizing":"border-box"
    },
    ".ag-popup-editor, .ag-popup-editor .ag-input-field-input, .ag-popup-editor input": {
        "font-size":"18px", "height":"36px", "line-height":"36px",
        "padding":"4px 10px", "width":"100%", "box-sizing":"border-box"
    },
    ".ag-cell.qty-edit.ag-cell-inline-editing": {
        "padding":"4px 8px !important"
    }
}


# Deprecation-safe call: prefer update_on if available
aggrid_kwargs = dict(
    gridOptions=grid_options,
    data_return_mode=DataReturnMode.AS_INPUT,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    theme="balham",
    height=640,
    custom_css=custom_css,
)
if "update_on" in signature(AgGrid).parameters:
    aggrid_kwargs["update_on"] = ["cellValueChanged"]
else:
    # kept for older versions (harmless if unused)
    from st_aggrid import GridUpdateMode
    aggrid_kwargs["update_mode"] = GridUpdateMode.VALUE_CHANGED

grid_resp = AgGrid(grid_df, **aggrid_kwargs)

# --- Unsaved quantity changes banner ---
try:
    _resp_df = None
    if isinstance(grid_resp, dict) and "data" in grid_resp:
        import pandas as _pd
        _resp_df = _pd.DataFrame(grid_resp["data"])
    elif hasattr(grid_resp, "data"):
        import pandas as _pd
        _resp_df = _pd.DataFrame(grid_resp.data)
    if _resp_df is not None and "Quantity" in _resp_df.columns:
        import pandas as _pd
        q_now = _pd.to_numeric(_resp_df.get("Quantity"), errors="coerce").fillna(0).astype(int)
        q_orig = _pd.to_numeric(_resp_df.get("OrigQuantity", 0), errors="coerce").fillna(0).astype(int)
        _changes = int((q_now != q_orig).sum())
        if _changes > 0:
            changes_ph.info(f":warning: Changed rows waiting to be saved: {_changes}")
        else:
            changes_ph.empty()
except Exception:
    pass
# Bottom actions
st.markdown("---")
c1,c2,c3 = st.columns(3)
save_clicked = c1.button("💾 Save edits to CSV", use_container_width=True)
c2.download_button("💾 Download filtered CSV (table)",
                   data=table_df.to_csv(sep=";", index=False).encode("utf-8"),
                   file_name="db_filtered.csv", mime="text/csv", use_container_width=True)
c3.download_button("⬇️ Download full CSV",
                   data=df.to_csv(sep=";", index=False).encode("utf-8"),
                   file_name="db.csv", mime="text/csv", use_container_width=True)

# Save edits (optional auto-merge by lookupID)
if save_clicked:
    # Get edited grid data robustly across st-aggrid versions
    edited_df = None
    try:
        if isinstance(grid_resp, dict) and "data" in grid_resp:
            edited_df = pd.DataFrame(grid_resp["data"])
        elif hasattr(grid_resp, "data"):
            edited_df = pd.DataFrame(grid_resp.data)  # fallback
    except Exception:
        edited_df = None
    if edited_df is None or edited_df.empty:
        edited_df = grid_df.copy()

    # Load LIVE CSV
    live = pd.read_csv(CSV_PATH, delimiter=";")

    # Ensure columns exist / types
    for col, typ in [("Quantity", int), ("Raw Price", float), ("PSA 10 Price", float), ("Raw Total", float)]:
        if col not in live.columns:
            live[col] = 0
    live["Quantity"]     = pd.to_numeric(live["Quantity"], errors="coerce").fillna(0).astype(int)
    live["Raw Price"]    = pd.to_numeric(live["Raw Price"], errors="coerce").fillna(0.0)
    if "PSA 10 Price" not in live.columns: live["PSA 10 Price"] = 0.0
    live["PSA 10 Price"] = pd.to_numeric(live["PSA 10 Price"], errors="coerce").fillna(0.0)
    live["Raw Total"]    = pd.to_numeric(live["Raw Total"], errors="coerce").fillna(0.0)

    edited_df["Quantity"] = pd.to_numeric(edited_df["Quantity"], errors="coerce").fillna(0).astype(int)

    updated_rows = 0

    # --- Primary match: lookupID (most stable) ---
    if "lookupID" in live.columns and "lookupID" in edited_df.columns:
        new_qty_map = dict(zip(edited_df["lookupID"].astype(str), edited_df["Quantity"].astype(int)))
        for idx, key in live["lookupID"].astype(str).items():
            if key in new_qty_map:
                new_q = int(new_qty_map[key])
                if int(live.at[idx, "Quantity"]) != new_q:
                    live.at[idx, "Quantity"] = new_q
                    try:
                        rp = float(live.at[idx, "Raw Price"])
                        live.at[idx, "Raw Total"] = round(new_q * rp, 2)
                    except Exception:
                        pass
                    updated_rows += 1
    else:
        # --- Fallback match: composite key of Set|Name|ID|Type ---
        def make_key(df):
            def col(c): return df[c].astype(str) if c in df.columns else ""
            return col("Set") + "|" + col("Name") + "|" + col("ID") + "|" + col("Type")

        live_key = make_key(live)
        edited_key = make_key(edited_df)
        new_qty_map = dict(zip(edited_key, edited_df["Quantity"].astype(int)))

        for idx, key in live_key.items():
            if key in new_qty_map:
                new_q = int(new_qty_map[key])
                if int(live.at[idx, "Quantity"]) != new_q:
                    live.at[idx, "Quantity"] = new_q
                    try:
                        rp = float(live.at[idx, "Raw Price"])
                        live.at[idx, "Raw Total"] = round(new_q * rp, 2)
                    except Exception:
                        pass
                    updated_rows += 1

    # Optional auto-merge
    if auto_merge:
        before = len(live)
        live = merge_by_lookupid(live)
        if len(live) < before:
            st.info(f"Auto-merge after edits: consolidated {before - len(live)} row(s) by lookupID.")

    # Save back to CSV
    live.to_csv(CSV_PATH, sep=";", index=False)
    st.success(f"Saved to db.csv — {updated_rows} row(s) updated.")
    st.rerun()

# Update prices / Merge buttons
if update_btn:
    save_rate(usd_to_gbp); st.session_state.saved_rate = usd_to_gbp
    if update_only_current_set:
        shown_keys = table_df['lookupID'].astype(str).unique().tolist()
        idxs = df.index[df['lookupID'].astype(str).isin(shown_keys)].tolist()
        # When forcing update for shown cards, treat as force for those rows
        effective_force = True
    else:
        # Apply to ALL rows
        idxs = df.index.tolist()
        effective_force = bool(force_update)
    st.info("Updating prices…")
    prog_text = st.empty()
    progress = st.progress(0)
    total = len(idxs)
    updated = 0
    for n,i in enumerate(idxs, start=1):
        row = df.loc[i]
        if effective_force or float(row["Raw Price"])==0.0 or float(row.get("PSA 10 Price",0))==0.0:
            det = fetch_pc_details(str(row["lookupID"]), usd_to_gbp)
            if effective_force or float(row["Raw Price"])==0.0:
                if det.get("price_gbp",0)>0:
                    df.at[i,"Raw Price"] = float(det["price_gbp"])
                    updated += 1
            # PSA10
            if "PSA 10 Price" not in df.columns: df["PSA 10 Price"] = 0.0
            if det.get("psa10_gbp",0) > 0 and (effective_force or float(df.at[i,"PSA 10 Price"])==0.0):
                df.at[i,"PSA 10 Price"] = float(det["psa10_gbp"])
        pct = int(n/max(total,1)*100)
        # Build a helpful label like "Fetching <name> 65/876"
        try:
            nm_tmp = str(df.at[i, "Name"]) if "Name" in df.columns else str(row.get("Name", ""))
        except Exception:
            nm_tmp = str(df.at[i, "lookupID"]) if "lookupID" in df.columns else ""
        label = f"Fetching {nm_tmp} {n}/{total}"
        try:
            # Streamlit >= 1.22 supports the 'text' kwarg
            progress.progress(pct, text=label)
        except TypeError:
            # Older Streamlit fallback: show text separately
            prog_text.write(label)
            progress.progress(pct)
    df["Raw Total"] = (pd.to_numeric(df["Quantity"], errors="coerce").fillna(0) *
                       pd.to_numeric(df["Raw Price"], errors="coerce").fillna(0)).round(2)
    if auto_merge:
        before=len(df)
        df = merge_by_lookupid(df)
        if len(df)<before:
            st.info(f"Auto-merge after update: consolidated {before-len(df)} row(s) by lookupID.")
    df.to_csv(CSV_PATH, sep=";", index=False)
    st.success(f"Updated {updated} row(s).")
    st.rerun()

if merge_btn:
    before=len(df)
    df = merge_by_lookupid(df)
    df.to_csv(CSV_PATH, sep=";", index=False)
    st.success(f"Merged duplicates by lookupID. Reduced {before - len(df)} row(s).")
    st.rerun()

# ---- Run image tasks AFTER the table is rendered so the grid stays visible ----
if fetch_missing_btn or refresh_broken_btn:
    if images_only_current_set and selected_set != "All sets":
        rows = df[df["Set"].astype(str) == selected_set]; scope = f"(only set: {selected_set})"
    else:
        rows = df; scope = "(all sets)"
    with st.status("Working on images…", expanded=True) as status:
        st.write(("Fetching" if fetch_missing_btn else "Refreshing") + f" images {scope}…")
        n = ensure_images_for_rows(rows)
        status.update(label=f"Done: processed {n} image(s).", state="complete", expanded=False)
    st.rerun()