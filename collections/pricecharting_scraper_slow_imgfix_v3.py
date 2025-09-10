#!/usr/bin/env python3
"""
pricecharting_scraper_slow_imgfix_v3.py
- Slow & polite PriceCharting scraper with robust image detection + Selenium cookies
- Adds: --only-id <N>  to process only the row whose config 'id' equals N

Your config.csv should include at least: id, Name, link
Delimiter is auto-detected (; , | or tab)

Examples
--------
# run everything (visible Chrome + verbose image logs)
py -u pricecharting_scraper_slow_imgfix_v3.py --config config.csv --cache cache --out . --debug-images

# run only config row id 2
py -u pricecharting_scraper_slow_imgfix_v3.py --config config.csv --cache cache --out . --only-id 2 --debug-images
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

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

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
IMG_ACCEPT = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")


def detect_delim(sample: str) -> str:
    cands = [";", ",", "\t", "|"]
    counts = {c: sample.count(c) for c in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ","


def read_config(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096); f.seek(0)
        delim = detect_delim(sample)
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            # normalize headers
            rid = (r.get("id") or r.get("ID") or r.get("Id") or "").strip()
            name = (r.get("Name") or r.get("name") or "").strip()
            link = (r.get("link") or r.get("url") or "").strip()
            if name and link:
                # try to coerce id to int if possible
                try:
                    rid_int: Optional[int] = int(rid)
                except Exception:
                    rid_int = None
                rows.append({"id": rid_int, "id_raw": rid, "Name": name, "link": link})
    if not rows:
        raise SystemExit("No rows found with both 'Name' and 'link' in config.")
    return rows


def slug_from_set_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    return parts[-1] if parts else ""


def is_card_href(href_path: str, set_slug: str) -> bool:
    return href_path.startswith(f"/game/{set_slug}/")


def normalize_lookupid(card_url: str) -> str:
    path = urlparse(card_url).path.strip("/")
    return path[len("game/"):] if path.startswith("game/") else path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def new_driver(headless: bool):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
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
        body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
        if "answer:" in body_text and "submit" in body_text:
            return True
    except Exception:
        pass
    return False


def wait_for_cards_or_human_check(driver, timeout: float):
    end = time.time() + timeout
    while time.time() < end:
        if driver.find_elements(By.CSS_SELECTOR, "a[href^='/game/']"):
            return "cards"
        if page_has_human_check(driver):
            return "human"
        time.sleep(0.3)
    return "timeout"


def gentle_collect_links(driver, set_slug: str, max_rounds: int = 220):
    seen = set()
    actions = ActionChains(driver)
    stagnant = 0
    last = -1
    for i in range(max_rounds):
        # collect
        for a in driver.find_elements(By.CSS_SELECTOR, "a[href^='/game/']"):
            href = a.get_attribute("href") or ""
            p = urlparse(href)
            if is_card_href(p.path, set_slug):
                seen.add(href.split("?")[0])

        # scroll & pause
        driver.execute_script("window.scrollBy(0, 700);")
        if i % 6 == 0:
            actions.key_down(Keys.PAGE_DOWN).pause(0.05).key_up(Keys.PAGE_DOWN).perform()
        time.sleep(0.8 + random.uniform(0, 0.6))

        curr = len(seen)
        if curr == last:
            stagnant += 1
        else:
            stagnant = 0
        last = curr
        if stagnant >= 8:
            break
    return sorted(seen)


def sync_cookies_from_driver(driver, session: requests.Session):
    for c in driver.get_cookies():
        name = c.get("name"); value = c.get("value")
        domain = c.get("domain") or "www.pricecharting.com"
        path = c.get("path") or "/"
        if name and value:
            try:
                session.cookies.set(name, value, domain=domain, path=path)
            except Exception:
                pass


def _normalize_url(u: str, base: str) -> str:
    if not u:
        return u
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return urljoin(base, u)
    return u


def _pick_from_srcset(srcset: str, base: str) -> str | None:
    best_url, best_w = None, -1
    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split()
        url = _normalize_url(bits[0], base)
        w = -1
        for b in bits[1:]:
            m = re.match(r"(\d+)w", b)
            if m:
                w = int(m.group(1))
                break
        if w == -1:
            w = 0
        if url and (url.endswith(IMG_EXTS) or "pricecharting" in url or "cloudfront" in url):
            if w > best_w:
                best_w, best_url = w, url
    return best_url


def extract_image_url_from_driver(driver) -> str | None:
    base = driver.current_url
    # 1) Direct <img> candidates
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
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in els:
            for attr in ["src", "data-src", "data-original", "data-lazy", "data-image"]:
                v = el.get_attribute(attr)
                if v:
                    url = _normalize_url(v, base)
                    if any(url.endswith(ext) for ext in IMG_EXTS) or "pricecharting" in url or "cloudfront" in url:
                        return url
            ss = el.get_attribute("srcset") or ""
            if ss:
                url = _pick_from_srcset(ss, base)
                if url:
                    return url

    # 2) <meta> fallbacks
    metas = driver.find_elements(By.CSS_SELECTOR, "meta[property='og:image'],meta[name='og:image'],meta[name='twitter:image']")
    for m in metas:
        v = m.get_attribute("content") or ""
        if v:
            return _normalize_url(v, base)

    # 3) <picture> sources
    for pic in driver.find_elements(By.CSS_SELECTOR, "picture source[srcset]"):
        ss = pic.get_attribute("srcset") or ""
        url = _pick_from_srcset(ss, base)
        if url:
            return url

    # 4) CSS background-image
    for el in driver.find_elements(By.CSS_SELECTOR, "[style*='background-image']"):
        style = el.get_attribute("style") or ""
        m = re.search(r"background-image\s*:\s*url\((['\"]?)(.+?)\1\)", style, re.I)
        if m:
            url = _normalize_url(m.group(2), base)
            if any(url.endswith(ext) for ext in IMG_EXTS) or "pricecharting" in url or "cloudfront" in url:
                return url

    # 5) Last-resort: regex in page HTML
    html = driver.page_source or ""
    m = re.search(r"https?://[^\"'>]+\.(?:jpg|jpeg|png|webp|gif)\b", html, re.I)
    if m:
        return m.group(0)

    return None


def download_image_with_cookies(img_url: str, referer: str, dest_path: Path,
                                session: requests.Session, debug: bool = False):
    try:
        headers = {
            "User-Agent": UA,
            "Referer": referer,
            "Accept": IMG_ACCEPT,
            "Accept-Language": "en-GB,en",
        }
        r = session.get(img_url, headers=headers, timeout=60, stream=True)
        if debug:
            print(f"[image] GET {img_url} -> {r.status_code}")
        r.raise_for_status()
        ext = os.path.splitext(urlparse(img_url).path)[1].lower() or ".jpg"
        if ext not in IMG_EXTS:
            ext = ".jpg"
        dest_path = dest_path.with_suffix(ext)
        ensure_dir(dest_path.parent)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
        return str(dest_path)
    except Exception as e:
        if debug:
            print(f"[image] Failed {img_url}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Slow scraper with robust image detection & cookie-aware downloads (filter by id).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=".")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--debug-images", action="store_true")
    ap.add_argument("--only-id", type=int, default=None, help="Process only the row whose config 'id' equals this number")
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
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img, meta[property='og:image'], meta[name='og:image'], meta[name='twitter:image']")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, "h1, .product-title, title"))
                    )
                )
                sync_cookies_from_driver(driver, sess)
                img_url = extract_image_url_from_driver(driver)

                # Parse set number
                set_number = ""
                try:
                    title_text = driver.title or ""
                    h1s = driver.find_elements(By.CSS_SELECTOR, "h1, .product-title")
                    if h1s:
                        title_text = h1s[0].text or title_text
                    m = re.search(r"#\s*(\d+)", title_text)
                    if m:
                        set_number = m.group(1)
                except Exception:
                    pass

                if img_url:
                    dest = Path(args.cache) / lookupid
                    saved = download_image_with_cookies(img_url, link, dest, sess, debug=args.debug_images)
                    if saved:
                        print(f"[image] saved -> {saved}")
                    else:
                        print(f"[image] FAILED -> {img_url}")
                else:
                    print("[image] No image element or meta image found on page")

                out_rows.append({"lookupid": lookupid, "set number": set_number})
                time.sleep(0.4 + random.uniform(0, 0.4))

            out_name = re.sub(r"\s+", "_", name.strip()) + ".csv"
            out_path = Path(args.out) / out_name
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "lookupid", "set number", "Availability"])
                for idx, r in enumerate(out_rows, 1):
                    w.writerow([idx, r["lookupid"], r.get("set number",""), ""])
            print(f"[write] Wrote {out_path} ({len(out_rows)} rows)")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
