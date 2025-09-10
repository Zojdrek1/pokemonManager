#!/usr/bin/env python3
"""
pricecharting_scraper_hires_v4.py
- Slow & polite PriceCharting scraper with robust image detection + Selenium cookies
- NEW: chooses the highest-resolution image via <img>/<picture> srcset
- NEW: optional --hires-tweak to try larger width values in query strings (w,width,h,height)
- Supports: --only-id N  (process just one row from config)

Run (visible Chrome + verbose image logs):
  py -u pricecharting_scraper_hires_v4.py --config config.csv --cache cache --out . --debug-images

Run only id=2:
  py -u pricecharting_scraper_hires_v4.py --config config.csv --cache cache --out . --only-id 2 --debug-images

Try query upscaling too:
  py -u pricecharting_scraper_hires_v4.py --config config.csv --cache cache --out . --only-id 2 --debug-images --hires-tweak
"""
from __future__ import annotations

import argparse, csv, os, re, sys, time, random
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, urljoin, parse_qsl, urlencode, urlunparse, unquote

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
IMG_ACCEPT = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")

# ---------------- CSV ----------------
def _detect_delim(sample: str) -> str:
    cands = [";", ",", "\t", "|"]
    counts = {c: sample.count(c) for c in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ","

def read_config(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096); f.seek(0)
        delim = _detect_delim(sample)
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            rid = (r.get("id") or r.get("ID") or r.get("Id") or "").strip()
            name = (r.get("Name") or r.get("name") or "").strip()
            link = (r.get("link") or r.get("url") or "").strip()
            if name and link:
                try:
                    rid_int: Optional[int] = int(rid)
                except Exception:
                    rid_int = None
                rows.append({"id": rid_int, "id_raw": rid, "Name": name, "link": link})
    if not rows:
        raise SystemExit("No rows found with both 'Name' and 'link' in config.")
    return rows

# --------------- utilities ---------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def slug_from_set_url(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    return parts[-1] if parts else ""

def is_card_href(href_path: str, set_slug: str) -> bool:
    return href_path.startswith(f"/game/{set_slug}/")

def normalize_lookupid(card_url: str) -> str:
    """
    Turn a card URL into the lookupid used in CSV and file paths.
    Decodes percent-encoding so e.g. ".../acerola%27s-mischief-84"
    becomes "acerola's-mischief-84".
    """
    path = urlparse(card_url).path or ""
    path = path.strip("/")
    if path.startswith("game/"):
        path = path[len("game/"):]
    # Decode %xx → characters (e.g., %27 → ')
    path = unquote(path, encoding="utf-8", errors="strict")
    return path

def _normalize_url(u: str, base: str) -> str:
    if not u: return u
    if u.startswith("//"): return "https:" + u
    if u.startswith("/"):  return urljoin(base, u)
    return u

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

# --------------- Selenium ---------------
def new_driver(headless: bool):
    opts = webdriver.ChromeOptions()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1100")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    opts.add_argument("--disable-notifications")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(2)
    return driver

def page_has_human_check(driver) -> bool:
    try:
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        return ("answer:" in body) and ("submit" in body)
    except Exception:
        return False

def wait_for_cards_or_human_check(driver, timeout: float):
    end = time.time() + timeout
    while time.time() < end:
        if driver.find_elements(By.CSS_SELECTOR, "a[href^='/game/']"): return "cards"
        if page_has_human_check(driver): return "human"
        time.sleep(0.3)
    return "timeout"

def gentle_collect_links(driver, set_slug: str, max_rounds: int = 220):
    seen = set(); actions = ActionChains(driver); last = -1; stagnant = 0
    for i in range(max_rounds):
        for a in driver.find_elements(By.CSS_SELECTOR, "a[href^='/game/']"):
            href = a.get_attribute("href") or ""
            if is_card_href(urlparse(href).path, set_slug):
                seen.add(href.split("?")[0])

        driver.execute_script("window.scrollBy(0, 700);")
        if i % 6 == 0:
            actions.key_down(Keys.PAGE_DOWN).pause(0.05).key_up(Keys.PAGE_DOWN).perform()
        time.sleep(0.8 + random.uniform(0, 0.6))

        curr = len(seen)
        stagnant = stagnant + 1 if curr == last else 0
        last = curr
        if stagnant >= 8: break
    return sorted(seen)

# ---------- image candidate collection ----------
def _pick_from_srcset(srcset: str, base: str) -> List[Tuple[str,int]]:
    out = []
    for part in (srcset or "").split(","):
        bits = part.strip().split()
        if not bits: continue
        url = _normalize_url(bits[0], base)
        w = 0
        for b in bits[1:]:
            m = re.match(r"(\d+)w", b)
            if m: w = int(m.group(1)); break
        if url:
            out.append((url, w))
    return out

def collect_image_candidates(driver) -> List[str]:
    """Return candidate image URLs, best first."""
    base = driver.current_url
    scored: List[Tuple[str,int]] = []

    # 1) direct <img> (src/data-*) and srcset
    img_selectors = [
        "img#product-image",
        "img.product-image",
        ".image-gallery img",
        ".gallery img",
        "img[alt^='Image:']",
        "img[src*='pricecharting']",
        "img[src*='cloudfront']",
    ]
    for sel in img_selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            # src/src-like
            for attr in ["src","data-src","data-original","data-lazy","data-image"]:
                v = el.get_attribute(attr)
                if v:
                    u = _normalize_url(v, base)
                    scored.append((u, 0))
            # srcset
            scored.extend(_pick_from_srcset(el.get_attribute("srcset") or "", base))

    # 2) <picture><source srcset>
    for pic in driver.find_elements(By.CSS_SELECTOR, "picture source[srcset]"):
        scored.extend(_pick_from_srcset(pic.get_attribute("srcset") or "", base))

    # 3) opengraph/twitter
    for m in driver.find_elements(By.CSS_SELECTOR, "meta[property='og:image'],meta[name='og:image'],meta[name='twitter:image']"):
        v = m.get_attribute("content") or ""
        if v: scored.append((_normalize_url(v, base), 0))

    # 4) CSS background-image
    for el in driver.find_elements(By.CSS_SELECTOR, "[style*='background-image']"):
        style = el.get_attribute("style") or ""
        mm = re.search(r"background-image\s*:\s*url\((['\"]?)(.+?)\1\)", style, re.I)
        if mm:
            scored.append((_normalize_url(mm.group(2), base), 0))

    # 5) last resort: regex in HTML
    html = driver.page_source or ""
    mm = re.findall(r"https?://[^\"'>]+\.(?:jpg|jpeg|png|webp|gif|avif)\b", html, flags=re.I)
    for u in mm:
        scored.append((u, 0))

    # prefer widest srcset weight, then keep order
    scored.sort(key=lambda t: t[1], reverse=True)
    return _dedupe_preserve_order([u for (u,_) in scored])

def tweak_query_for_hires(url: str, max_w: int = 1600, max_h: int = 1600) -> str:
    """If URL has width/height hints, try bumping them up."""
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    changed = False
    for k in list(q.keys()):
        lk = k.lower()
        if lk in ("w","width"):  q[k] = str(max_w); changed = True
        if lk in ("h","height"): q[k] = str(max_h); changed = True
        # common generic param like "s=300"
        if lk in ("s","size"):
            try:
                int(q[k]); q[k] = str(max(max_w, max_h)); changed = True
            except Exception:
                pass
    if changed:
        return urlunparse(p._replace(query=urlencode(q, doseq=True)))
    # Also try stripping explicit small size suffixes like "-200x200" (very generic; only on filename end)
    m = re.search(r"-(\d+)x(\d+)(\.[a-z]+)$", p.path, re.I)
    if m:
        new_path = re.sub(r"-(\d+)x(\d+)(\.[a-z]+)$", r"\3", p.path, flags=re.I)
        return urlunparse(p._replace(path=new_path))
    return url

# --------------- download ---------------
def sync_cookies_from_driver(driver, session: requests.Session):
    for c in driver.get_cookies():
        name, value = c.get("name"), c.get("value")
        domain = c.get("domain") or "www.pricecharting.com"
        path = c.get("path") or "/"
        if name and value:
            try: session.cookies.set(name, value, domain=domain, path=path)
            except Exception: pass

def try_download_first_ok(candidates: List[str], referer: str, dest_base: Path,
                          session: requests.Session, debug=False) -> Optional[str]:
    ensure_dir(dest_base.parent)
    for u in candidates:
        try:
            headers = {"User-Agent": UA, "Referer": referer, "Accept": IMG_ACCEPT, "Accept-Language": "en-GB,en"}
            r = session.get(u, headers=headers, timeout=60, stream=True)
            if debug: print(f"[image] GET {u} -> {r.status_code}")
            r.raise_for_status()
            # decide extension
            ext = os.path.splitext(urlparse(u).path)[1].lower() or ".jpg"
            if ext not in IMG_EXTS: ext = ".jpg"
            dest = dest_base.with_suffix(ext)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk: f.write(chunk)
            return str(dest)
        except Exception as e:
            if debug: print(f"[image] FAIL {u}: {e}")
            continue
    return None

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser(description="Hi-res PriceCharting scraper (robust image pick + optional query upsize, filter by id).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=".")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--debug-images", action="store_true")
    ap.add_argument("--only-id", type=int, default=None, help="Process only the row whose config 'id' equals this number")
    ap.add_argument("--hires-tweak", action="store_true", help="Try bumping width/height query params for an even larger image")
    args = ap.parse_args()

    rows = read_config(args.config)
    if args.only_id is not None:
        rows = [r for r in rows if r["id"] == args.only_id]
        if not rows:
            print(f"[info] No config row with id == {args.only_id}. Nothing to do.")
            return

    driver = new_driver(headless=args.headless)
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept-Language": "en-GB,en"})

    try:
        for row in rows:
            name = row["Name"]; set_url = row["link"]
            set_slug = slug_from_set_url(set_url)

            print(f"\n=== {name} (config id: {row['id_raw']}) ===")
            driver.get(set_url)
            status = wait_for_cards_or_human_check(driver, timeout=25)
            if status == "human":
                print("\n[attention] Human-check detected. Please solve in Chrome and click SUBMIT.")
                input("Press Enter AFTER it's solved and card links are visible... ")

            links = gentle_collect_links(driver, set_slug)
            print(f"[collect] {len(links)} card links")

            out_rows = []
            for link in links:
                lookupid = normalize_lookupid(link)
                driver.get(link)
                WebDriverWait(driver, 20).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img, picture source[srcset], meta[property='og:image'], meta[name='og:image'], meta[name='twitter:image']")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "h1, .product-title, title"))
                    )
                )
                # sync cookies for CDN
                sync_cookies_from_driver(driver, sess)

                # gather candidates and prefer biggest
                candidates = collect_image_candidates(driver)
                # optional: add "tweaked" hires variants
                if args.hires_tweak:
                    tweaked = [tweak_query_for_hires(u) for u in candidates]
                    # put tweaked first so we try them before originals
                    candidates = _dedupe_preserve_order(tweaked + candidates)

                # download first that works
                dest_base = Path(args.cache) / lookupid
                saved = try_download_first_ok(candidates, referer=link, dest_base=dest_base, session=sess, debug=args.debug_images)
                if saved:
                    print(f"[image] saved -> {saved}")
                else:
                    print("[image] FAILED (no candidate worked)")

                # set number
                set_number = ""
                try:
                    title_text = driver.title or ""
                    h1s = driver.find_elements(By.CSS_SELECTOR, "h1, .product-title")
                    if h1s: title_text = h1s[0].text or title_text
                    m = re.search(r"#\s*(\d+)", title_text)
                    if m: set_number = m.group(1)
                except Exception:
                    pass

                out_rows.append({"lookupid": lookupid, "set number": set_number})
                time.sleep(0.4 + random.uniform(0, 0.4))  # polite delay

            # write CSV
            out_name = re.sub(r"\s+", "_", name.strip()) + ".csv"
            out_path = Path(args.out) / out_name
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "lookupid", "set number", "Availability"])
                for idx, r in enumerate(out_rows, 1):
                    w.writerow([idx, r["lookupid"], r.get("set number",""), ""])
            print(f"[write] Wrote {out_path} ({len(out_rows)} rows)")
    finally:
        try: driver.quit()
        except Exception: pass

if __name__ == "__main__":
    main()
