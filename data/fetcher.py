#!/usr/bin/env python
# coding: utf-8

"""
Modified version of fetcher.py that uses Playwright with stealth to avoid
basketball-reference.com's anti-botting measures. The original requests-based
approach returned 403 for all team game log pages. Playwright, combined with
the playwright‑stealth plugin, mimics a real browser and bypasses these blocks.

Requirements:
  pip install requests beautifulsoup4 pandas tqdm playwright
  pip install playwright-stealth
  playwright install chromium

  (These must be installed in your environment.)

The changes are concentrated in:
  - _get_page_html_async()    : launches a stealthy Chromium context, fetches the page, and returns its HTML.
  - fetch_with_cache()        : now calls _get_page_html_async() via asyncio.run() instead of requests.
  - The rest of the code (parsing, CSV handling, etc.) remains identical to the original notebook.
"""

import asyncio
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import bs4
import pandas as pd
from tqdm import tqdm

# ---------- Playwright & Stealth ----------
try:
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth
except ImportError:
    raise ImportError(
        "Please install the required packages:\n"
        "  pip install playwright playwright-stealth\n"
        "  playwright install chromium"
    )

# ---------- Original Team Codes ----------
team_codes: Dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# ---------- Minimal Politeness ----------
def polite_delay(min_s: float = 2.0, max_s: float = 5.0) -> None:
    """Randomised delay to mimic human browsing speed."""
    time.sleep(random.uniform(min_s, max_s))

# ---------- Core Playwright Fetcher ----------
async def _get_page_html_async(
    url: str,
    headless: bool = True,
    retries: int = 3,
    viewport: Optional[Dict[str, int]] = None,
) -> str:
    """
    Launch a stealth‑patched Chromium browser, navigate to *url*, wait for the
    table to appear, and return the full HTML content.

    Parameters
    ----------
    url : str
        The URL to fetch.
    headless : bool
        Run the browser in headless mode (True) or with a visible window (False).
        Headless is faster and more server‑friendly, but a visible window can help
        debugging.
    retries : int
        Number of times to retry on failure (e.g., timeout, network issue).
    viewport : dict | None
        Browser window size, e.g. {"width": 1920, "height": 1080}.

    Returns
    -------
    str
        The rendered HTML source of the page.
    """
    if viewport is None:
        viewport = {"width": 1920, "height": 1080}

    for attempt in range(1, retries + 1):
        try:
            # Use the Stealth context manager to apply all evasion patches
            async with Stealth().use_async(async_playwright()) as playwright:
                browser = await playwright.chromium.launch(
                    headless=headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                    ],
                )
                context = await browser.new_context(
                    viewport=viewport,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                # Wait for the main table to appear (id="team_game_log_reg" or similar)
                await page.wait_for_selector(
                    "table#team_game_log_reg, table#team_game_log_adv_reg",
                    timeout=30000,
                )
                # Extra small random wait to let stats fully load
                await asyncio.sleep(random.uniform(0.5, 1.5))
                html = await page.content()
                await browser.close()
                return html
        except Exception as exc:
            print(f"  [Attempt {attempt}/{retries}] Error fetching {url}: {exc}")
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)  # Exponential back‑off
            else:
                raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from exc
    # Unreachable, but required by type checker
    return ""

# ---------- Caching Wrapper (unchanged API) ----------
def fetch_with_cache(url: str, cache_path: Optional[str] = None) -> bytes:
    """
    Fetch the URL with optional local caching.

    Unlike the original version, this function now retrieves the page content
    via Playwright (stealth).  If a cache file exists, it is used directly.

    Parameters
    ----------
    url : str
        The URL to fetch.
    cache_path : str | None
        File path for cached content. If None, caching is disabled.

    Returns
    -------
    bytes
        The raw HTML content (UTF‑8 encoded).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    # Run the async Playwright function synchronously
    html = asyncio.run(_get_page_html_async(url))
    polite_delay()  # extra politeness between sequential fetches

    content = html.encode("utf-8")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(content)

    return content

# ---------- HTML Helpers (unchanged) ----------
def uncomment_hidden_tables(html_bytes: bytes) -> str:
    """Remove HTML comments to reveal hidden tables (e.g., playoffs)."""
    html_str = html_bytes.decode("utf-8", errors="replace")
    html_str = re.sub(r"<!--.*?-->", "", html_str, flags=re.DOTALL)
    return html_str

def parse_row(tr: bs4.element.Tag) -> Dict[str, str]:
    """Extract all statistical fields from a table row."""
    return {td["data-stat"]: td.text for td in tr.find_all("td")}

# ---------- Main Function (unchanged logic) ----------
def fetch_team_season_log(
    team: str, season: str, use_cache: bool = False
) -> Dict[str, Dict[str, Dict[str, Union[int, float]]]]:
    """
    Fetch all regular‑season and playoff game logs for a team.

    This version uses Playwright to obtain the HTML, but the parsing logic
    remains identical to the original notebook.

    Parameters
    ----------
    team : str
    season : str
    use_cache : bool

    Returns
    -------
    dict
    """
    basic_url = f"https://www.basketball-reference.com/teams/{team_codes[team]}/{season}/gamelog/"
    advanced_url = f"https://www.basketball-reference.com/teams/{team_codes[team]}/{season}/gamelog-advanced/"
    cache_base = f"cache/{team}_{season}"

    basic_html = fetch_with_cache(basic_url, f"{cache_base}_basic.html" if use_cache else None)
    advanced_html = fetch_with_cache(advanced_url, f"{cache_base}_adv.html" if use_cache else None)

    basic_soup = bs4.BeautifulSoup(uncomment_hidden_tables(basic_html), "html.parser")
    advanced_soup = bs4.BeautifulSoup(uncomment_hidden_tables(advanced_html), "html.parser")

    basic_logs = basic_soup.find_all(
        "tr", id=lambda x: x and (x.startswith("team_game_log_reg.") or x.startswith("team_game_log_post."))
    )
    advanced_logs = advanced_soup.find_all(
        "tr", id=lambda x: x and (x.startswith("team_game_log_adv_reg.") or x.startswith("team_game_log_adv_post."))
    )

    stats = defaultdict(lambda: {"stats": {}, "average_stats": {}})

    for log in basic_logs:
        row = parse_row(log)
        date = row.get("date")
        if not date:
            continue
        stats[date]["stats"] = {
            "pts": int(row["team_game_score"]),
            "fg": int(row["fg"]),
            "fga": int(row["fga"]),
            "fg_pct": float(row["fg_pct"]),
            "fg3": int(row["fg3"]),
            "fg3a": int(row["fg3a"]),
            "fg3_pct": float(row["fg3_pct"]),
            "fg2": int(row["fg2"]),
            "fg2a": int(row["fg2a"]),
            "fg2_pct": float(row["fg2_pct"]),
            "ft": int(row["ft"]),
            "fta": int(row["fta"]),
            "ft_pct": float(row["ft_pct"]),
            "orb": int(row["orb"]),
            "drb": int(row["drb"]),
            "trb": int(row["trb"]),
            "ast": int(row["ast"]),
            "stl": int(row["stl"]),
            "blk": int(row["blk"]),
            "tov": int(row["tov"]),
            "pf": int(row["pf"]),
        }

    for log in advanced_logs:
        row = parse_row(log)
        date = row.get("date")
        if not date:
            continue
        stats[date]["stats"].update({
            "ortg": float(row["team_off_rtg"]),
            "drtg": float(row["team_def_rtg"]),
            "pace": float(row["pace"]),
            "ftr": float(row["fta_per_fga_pct"]),
            "3ptar": float(row["fg3a_per_fga_pct"]),
            "ts": float(row["ts_pct"]),
            "trb_pct": float(row["team_trb_pct"]),
            "ast_pct": float(row["team_ast_pct"]),
            "stl_pct": float(row["team_stl_pct"]),
            "blk_pct": float(row["team_blk_pct"]),
            "efg_pct": float(row["efg_pct"]),
            "tov_pct": float(row["team_tov_pct"]),
            "orb_pct": float(row["team_orb_pct"]),
            "ft_rate": float(row["ft_rate"]),
        })

    return stats

# ---------- CSV Helpers (unchanged) ----------
def save_team_stats_to_csv(
    stats: Dict[str, Dict[str, Dict[str, Union[int, float]]]], team: str, file_path: str
) -> None:
    columns = [
        "date", "team", "pts", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
        "fg2", "fg2a", "fg2_pct", "ft", "fta", "ft_pct", "orb", "drb", "trb",
        "ast", "stl", "blk", "tov", "pf", "ortg", "drtg", "pace", "ftr", "3ptar",
        "ts", "trb_pct", "ast_pct", "stl_pct", "blk_pct", "efg_pct", "tov_pct",
        "orb_pct", "ft_rate",
    ]

    rows = []
    for date, data in stats.items():
        row = {"date": date, "team": team}
        row.update(data["stats"])
        rows.append(row)

    new_data = pd.DataFrame(rows).reindex(columns=columns, fill_value=None)

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path).reindex(columns=columns, fill_value=None)
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates(
            subset=["date", "team"], keep="first"
        )
    else:
        combined_data = new_data

    combined_data.to_csv(file_path, index=False)

def sort_csv() -> None:
    df = pd.read_csv("./csv/gamelogs.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["team", "date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv("./csv/gamelogs.csv", index=False)

# ---------- Main Entry Point ----------
if __name__ == "__main__":
    target_year = "2026"

    print(f"\nFetching NBA game logs for {target_year} season...\n")

    for team in tqdm(team_codes.keys(), desc="Overall progress", unit="team", colour="green"):
        stats = fetch_team_season_log(team, target_year)
        save_team_stats_to_csv(stats, team, "csv/gamelogs.csv")

    print("\nSorting CSV file...")
    sort_csv()
    print("All done! Data saved to csv/gamelogs.csv\n")