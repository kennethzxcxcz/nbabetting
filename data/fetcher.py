#!/usr/bin/env python
# coding: utf-8

# #### Fetching and Processing Team Season Logs
# 
# In this section, we define functions to scrape and process game log data for NBA teams from Basketball Reference. This includes both basic and advanced statistics for each game of a given season.
# 
# ##### Overview of the Data Retrieval Process
# 1. **Fetching Game Logs**
#    - The `fetch_team_season_log` function requests the game log pages for a specified team and season.
#    - It extracts both basic and advanced game statistics, including shooting percentages, turnovers, and efficiency metrics.
#    - The data is stored in a structured format, organized by game date.
# 
# 2. **Calculating Rolling Averages**
#    - To provide a better understanding of team performance trends, rolling averages of key statistics are computed.
#    - The most recent games have a higher weight in the rolling average calculation.
# 
# 3. **Saving Data to CSV**
#    - The `save_team_stats_to_csv` function ensures that fetched statistics are stored in a CSV file.
#    - The function avoids overwriting existing data and removes duplicates to maintain consistency.
# 
# 4. **Sorting and Organizing Data**
#    - The `sort_csv` function sorts the dataset by team and game date to improve readability and usability.
# 
# Finally, the script iterates through all teams, fetching and saving their statistics for the specified season before sorting the final dataset.

# In[1]:


# Importing libraries
import requests
import time
import random
import bs4
from collections import defaultdict
import pandas as pd
import os
from tqdm import tqdm

# Storing the teams names - codes pair
team_codes: dict[str, str] = {
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

# --- Session setup (reuse connections) ---
session: requests.Session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
})


def polite_delay(min_s: float = 1.5, max_s: float = 3.5) -> None:
    """
    Pause execution for a random duration within a specified range.

    This function introduces a randomized delay between HTTP requests to
    reduce the likelihood of triggering rate-limiting or anti-scraping
    mechanisms on the target website.

    Parameters
    ----------
    min_s : float, optional
        Minimum number of seconds to wait. Default is 1.5.
    max_s : float, optional
        Maximum number of seconds to wait. Default is 3.5.

    Returns
    -------
    None
        This function does not return a value.
    """
    time.sleep(random.uniform(min_s, max_s))


def get_with_retry(url: str, retries: int = 3) -> requests.models.Response:
    """
    Perform an HTTP GET request with retry and exponential backoff.

    This function attempts to retrieve a URL multiple times in case of
    temporary failures (e.g., rate limiting or server issues). Between
    attempts, it applies exponential backoff to reduce request pressure.

    Parameters
    ----------
    url : str
        The URL to request.
    retries : int, optional
        The maximum number of retry attempts. Default is 3.

    Returns
    -------
    requests.models.Response
        The successful HTTP response object.

    Raises
    ------
    requests.RequestException
        If all retry attempts fail.
    """
    for attempt in range(retries):
        response: requests.models.Response = session.get(url)

        if response.status_code == 200:
            return response

        time.sleep(2 ** attempt)

    response.raise_for_status()


def fetch_with_cache(url: str, cache_path: str | None) -> bytes:
    """
    Fetch a URL with optional local caching.

    This function retrieves the content of a URL, either by loading it from
    a local cache file (if available) or by performing an HTTP request.
    When caching is enabled, downloaded content is stored for reuse in
    subsequent runs.

    Parameters
    ----------
    url : str
        The URL to fetch.
    cache_path : str | None
        File path for cached content. If None, caching is disabled.

    Returns
    -------
    bytes
        The raw content of the HTTP response or cached file.

    Raises
    ------
    requests.RequestException
        If the HTTP request fails and no cached file is available.
    OSError
        If the cache file cannot be read or written.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    response: requests.models.Response = get_with_retry(url)
    polite_delay()

    content: bytes = response.content

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(content)

    return content


def parse_row(tr: bs4.element.Tag) -> dict[str, str]:
    """
    Extract all statistical fields from a table row.

    This function parses a BeautifulSoup <tr> element and converts all
    contained <td> elements into a dictionary keyed by their
    "data-stat" attributes.

    Parameters
    ----------
    tr : bs4.element.Tag
        A BeautifulSoup table row element.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each "data-stat" attribute to its
        corresponding text value.
    """
    return {td["data-stat"]: td.text for td in tr.find_all("td")}


def uncomment_hidden_tables(html_bytes: bytes) -> str:
    """
    Uncomment hidden tables in Basketball-Reference HTML.

    Basketball-Reference wraps playoff tables inside HTML comment blocks
    to lazy-load them. The comments use specific newline patterns:
    backslash-n<!--backslash-n and backslash-n-->backslash-n.
    This function strips those markers so that all tables become visible
    to BeautifulSoup.

    Parameters
    ----------
    html_bytes : bytes
        Raw HTML response content.

    Returns
    -------
    str
        HTML string with comment markers removed, exposing hidden tables.
    """
    html_str: str = html_bytes.decode("utf-8", errors="replace")
    # Basketball-Reference uses newline-wrapped comment markers to hide tables
    html_str = html_str.replace("\n<!--\n", "\n").replace("\n-->\n", "\n")
    # Also handle variations without newlines just in case
    html_str = html_str.replace("<!--", "").replace("-->", "")
    return html_str


# --- Main function ---
def fetch_team_season_log(
    team: str, season: str, use_cache: bool = True
) -> dict[str, dict[str, dict[str, int | float]]]:
    """
    Fetch all regular-season and playoff game logs for a team with caching and retry logic.

    This function downloads and parses both the basic and advanced team game
    log pages from Basketball-Reference. It merges per-game statistics into
    a unified structure keyed by game date, including playoff games that are
    hidden inside HTML comments on the page.

    To improve efficiency and reduce server load, the function supports:
    - Persistent HTTP sessions
    - Retry logic with exponential backoff
    - Optional local caching of HTML responses
    - Randomized delays between requests

    Parameters
    ----------
    team : str
        Team name key used to resolve the Basketball-Reference team code
        via the `team_codes` mapping.
    season : str
        NBA season year (e.g. "2024" for the 2023-24 season).
    use_cache : bool, optional
        Whether to cache downloaded HTML pages locally. Default is True.

    Returns
    -------
    dict[str, dict[str, dict[str, int | float]]]
        A dictionary keyed by game date. Each date contains:
        - "stats": a dictionary of merged basic and advanced statistics
        - "average_stats": an empty dictionary reserved for future use

    Raises
    ------
    KeyError
        If the provided team is not present in the `team_codes` mapping.
    ValueError
        If numeric fields cannot be converted due to unexpected formatting.
    requests.RequestException
        If HTTP requests fail after all retry attempts.
    OSError
        If cached files cannot be read or written.

    Notes
    -----
    - Both regular-season and playoff games are included.
    - This function depends on the current HTML structure of
      Basketball-Reference and may break if the layout changes.
    - Cached files are stored under the `cache/` directory.
    """

    basic_url: str = (
        f"https://www.basketball-reference.com/teams/{team_codes[team]}/{season}/gamelog/"
    )
    advanced_url: str = (
        f"https://www.basketball-reference.com/teams/{team_codes[team]}/{season}/gamelog-advanced/"
    )

    cache_base: str = f"cache/{team}_{season}"

    basic_html: bytes = fetch_with_cache(
        basic_url, f"{cache_base}_basic.html" if use_cache else None
    )
    advanced_html: bytes = fetch_with_cache(
        advanced_url, f"{cache_base}_adv.html" if use_cache else None
    )

    # Uncomment hidden playoff tables before parsing
    basic_html_uncommented: str = uncomment_hidden_tables(basic_html)
    advanced_html_uncommented: str = uncomment_hidden_tables(advanced_html)

    basic_soup: bs4.BeautifulSoup = bs4.BeautifulSoup(basic_html_uncommented, "html.parser")
    advanced_soup: bs4.BeautifulSoup = bs4.BeautifulSoup(
        advanced_html_uncommented, "html.parser"
    )

    # Find both regular season and playoff rows
    # Regular season rows have IDs like team_game_log_reg.N
    # Playoff rows have IDs like team_game_log_post.N
    basic_logs: list[bs4.element.Tag] = basic_soup.find_all(
        "tr", id=lambda x: x and (x.startswith("team_game_log_reg.") or x.startswith("team_game_log_post."))
    )
    advanced_logs: list[bs4.element.Tag] = advanced_soup.find_all(
        "tr", id=lambda x: x and (x.startswith("team_game_log_adv_reg.") or x.startswith("team_game_log_adv_post."))
    )

    stats: dict[str, dict[str, dict[str, int | float]]] = defaultdict(
        lambda: {"stats": {}, "average_stats": {}}
    )

    for log in basic_logs:
        row: dict[str, str] = parse_row(log)
        date: str | None = row.get("date")
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
        row: dict[str, str] = parse_row(log)
        date: str | None = row.get("date")
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


# Saves the fetched team stats into a CSV file without overwriting existing data,
# ensuring the format remains consistent and duplicates are avoided.
def save_team_stats_to_csv(
    stats: dict[str, dict[str, dict[str, int | float]]], team: str, file_path: str
) -> None:
    """
    Save a team's per-game season statistics to a CSV file.

    This function converts a dictionary of game-by-game statistics into a
    tabular format and appends it to a CSV file. If the file already exists,
    existing data is preserved and duplicate entries (based on team and date)
    are removed.

    The CSV schema is enforced to ensure consistent column ordering and
    presence across writes. Missing statistics are filled with NaN values.

    Parameters
    ----------
    stats : dict
        A dictionary of team statistics keyed by game date. Each date is
        expected to map to a dictionary containing a "stats" key with
        basic and advanced per-game metrics.
    team : str
        The team identifier to associate with each row in the CSV.
    file_path : str
        Path to the CSV file where the data will be saved. If the file
        exists, new data is merged with existing data.

    Returns
    -------
    None
        This function does not return a value. It writes data directly
        to a CSV file.

    Raises
    ------
    KeyError
        If a game entry in `stats` does not contain the expected "stats" key.
    pandas.errors.EmptyDataError
        If an existing CSV file is empty or malformed.
    OSError
        If the file cannot be read from or written to disk.

    Notes
    -----
    - Duplicate rows are identified using the combination of "date" and "team".
    - Column consistency is enforced even if new or existing data is missing
      some statistics.
    - The CSV file is overwritten with the merged dataset on each call.
    """

    # Define expected columns
    columns: list[str] = [
        "date",
        "team",
        "pts",
        "fg",
        "fga",
        "fg_pct",
        "fg3",
        "fg3a",
        "fg3_pct",
        "fg2",
        "fg2a",
        "fg2_pct",
        "ft",
        "fta",
        "ft_pct",
        "orb",
        "drb",
        "trb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "ortg",
        "drtg",
        "pace",
        "ftr",
        "3ptar",
        "ts",
        "trb_pct",
        "ast_pct",
        "stl_pct",
        "blk_pct",
        "efg_pct",
        "tov_pct",
        "orb_pct",
        "ft_rate",
    ]

    # Convert stats dictionary into a DataFrame
    rows: list = list()
    for date, data in stats.items():
        row: dict[str, str] = {"date": date, "team": team}
        row.update(data["stats"])  # Add raw stats
        rows.append(row)

    new_data: pd.DataFrame = pd.DataFrame(rows)

    # Ensure all required columns exist and fill missing ones with NaN
    new_data: pd.DataFrame = new_data.reindex(columns=columns, fill_value=None)

    # Check if file exists
    if os.path.exists(file_path):
        existing_data: pd.DataFrame = pd.read_csv(file_path)

        # Ensure column consistency
        existing_data: pd.DataFrame = existing_data.reindex(
            columns=columns, fill_value=None
        )

        # Remove duplicates based on team and date
        combined_data: pd.DataFrame = pd.concat(
            [existing_data, new_data]
        ).drop_duplicates(subset=["date", "team"], keep="first")
    else:
        combined_data: pd.DataFrame = new_data  # No existing data, write new data

    # Save to CSV
    combined_data.to_csv(file_path, index=False)


# Sort the dataframe for better organization
def sort_csv() -> None:
    """
    Sort the team game logs CSV file by team and date.

    This function reads the `gamelogs.csv` file, converts the date column
    to a datetime object for proper chronological sorting, and then sorts
    the data first by team name (A-Z) and then by game date (oldest to newest).
n    The sorted data is written back to the same CSV file.

    Returns
    -------
    None
        This function does not return a value. It modifies the CSV file
        in place.

    Raises
    ------
    FileNotFoundError
        If the `gamelogs.csv` file does not exist at the expected path.
    pandas.errors.ParserError
        If the CSV file cannot be parsed correctly.
    KeyError
        If the required "date" or "team" columns are missing.

    Notes
    -----
    - Dates are temporarily converted to datetime for sorting accuracy
      and then restored to `YYYY-MM-DD` string format.
    - The original file is overwritten with the sorted data.
    """

    # Read the CSV file into a DataFrame
    df: pd.DataFrame = pd.read_csv("./csv/gamelogs.csv")

    # Ensure date column is treated as datetime for sorting
    df["date"] = pd.to_datetime(df["date"])

    # Sort by home_team (A-Z) first, then by date
    df: pd.DataFrame = df.sort_values(by=["team", "date"])

    # Convert the date column back to its original format (YYYY-MM-DD)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Save the sorted DataFrame back to a CSV file
    df.to_csv("./csv/gamelogs.csv", index=False)


if __name__ == "__main__":
    target_year: str = "2026"

    # Main progress bar for all teams
    print(f"\nFetching NBA game logs for {target_year} season...\n")

    for team in tqdm(
        team_codes.keys(), desc="Overall progress", unit="team", colour="green"
    ):
        stats: dict[str, dict[str, dict[str, int | float]]] = fetch_team_season_log(
            team, target_year
        )
        save_team_stats_to_csv(stats, team, "csv/gamelogs.csv")

    print("\nSorting CSV file...")
    sort_csv()
    print("All done! Data saved to csv/gamelogs.csv\n")


# #### Fetching the Current Month's NBA Schedule
# 
# This function, `fetch_month_schedule`, retrieves the NBA game schedule for the current month from Basketball-Reference and appends new games to a CSV file, avoiding duplicates.
# 
# ##### How It Works:
# 1. **Determine the Current Month:**
#    - The function extracts the current month and formats it in lowercase to match the Basketball-Reference URL pattern.
# 
# 2. **Fetch the Schedule Page:**
#    - Constructs the URL dynamically based on the provided year.
#    - Sends a request to retrieve the webpage.
#    - Parses the HTML using BeautifulSoup.
# 
# 3. **Extract Game Data:**
#    - Finds all game rows within the table body.
#    - Extracts relevant details: game date, home team, and away team.
#    - Stops processing if a game has remarks (e.g., postponed or canceled).
# 
# 4. **Check for Existing Data:**
#    - Reads the existing CSV file to collect already stored games.
#    - Compares new games with existing entries to avoid duplicates.
# 
# 5. **Append New Data (If Any):**
#    - Writes only new rows to the CSV file.
#    - Displays a message indicating the number of new rows added or if no new games were found.
# 
# This ensures the latest schedule is fetched and updated in `./csv/schedule.csv`.

# In[2]:


# schedule.py – now loops over October → April
import datetime
import requests
import bs4
import csv
from tqdm import tqdm

MONTHS = ["october", "november", "december", "january", "february", "march", "april",'may','june']

def fetch_month_schedule(year: str, filename: str = "./csv/schedule.csv") -> None:
    for month in MONTHS:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
        try:
            tqdm.write(f"Fetching {month.capitalize()} {year} schedule...")
            page = requests.get(url)
            page.raise_for_status()
        except requests.HTTPError:
            tqdm.write(f"  ➜ Month {month} not available (maybe no games), skipped.")
            continue

        soup = bs4.BeautifulSoup(page.content, "html.parser")
        tbody = soup.find("tbody")
        if not tbody:
            tqdm.write(f"  ➜ No games for {month}, skipping.")
            continue

        rows = tbody.find_all("tr")
        games = []
        for row in tqdm(rows, desc=f"Processing {month}", unit="game", leave=False):
            date_th = row.find("th", {"data-stat": "date_game"})
            if not date_th:
                continue
            game = {
                "date": datetime.datetime.strptime(
                    date_th.text, "%a, %b %d, %Y"
                ).strftime("%Y-%m-%d"),
                "home_team": row.find("td", {"data-stat": "home_team_name"}).text,
                "away_team": row.find("td", {"data-stat": "visitor_team_name"}).text,
            }
            games.append(game)

        # Read existing data to avoid duplicates
        existing = set()
        try:
            with open(filename, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add((row["date"], row["home_team"], row["away_team"]))
        except FileNotFoundError:
            pass

        filtered = [g for g in games if (g["date"], g["home_team"], g["away_team"]) not in existing]
        if not filtered:
            tqdm.write(f"  ➜ No new games for {month}.")
            continue

        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team"])
            if not existing:  # new file → write header
                writer.writeheader()
            writer.writerows(filtered)
        tqdm.write(f"  ✓ Appended {len(filtered)} new games from {month}.")

if __name__ == "__main__":
    target_year = "2026"   # or derive from current date
    fetch_month_schedule(target_year)


# #### Formatting and Saving a CSV Schedule
# 
# This function processes a given CSV string containing basketball game results and formats it into a structured CSV file. The goal is to extract relevant game details such as date, teams, and scores, and determine the winner before saving the formatted data to a specified file.
# 
# ##### How It Works:
# 1. **Read the CSV String**: The function takes an input CSV string and splits it into rows.
# 2. **Process the Data**: Using a CSV reader, it skips the header row and iterates over each row to extract key columns:
#    - Date of the game
#    - Home team
#    - Away team
#    - Points scored by both teams
# 3. **Determine the Winner**: It checks which team has the higher score and assigns:
#    - `0` if the home team wins
#    - `1` if the away team wins
# 4. **Reformat the Date**: Converts the date format from `Tue Oct 24 2023` to `2023-10-24`.
# 5. **Save to File**: Writes the formatted data into a CSV file in append mode.
# 
# ##### Example Input CSV String:
# ```
# Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
# Tue Oct 24 2023,7:30p,Los Angeles Lakers,107,Denver Nuggets,119,Box Score,,19842,2:17,Ball Arena,
# ```
# 
# ##### Expected Output in `results.csv`:
# ```
# 2023-10-24,Denver Nuggets,Los Angeles Lakers,0
# ```

# In[3]:


import csv
import datetime
import time
import random
import requests
import bs4
from tqdm import tqdm

MONTHS = ["october", "november", "december", "january", "february", "march", "april"]
YEAR = "2026"  # Change to the current season year, or derive from datetime
OUTPUT_FILE = "./csv/results.csv"

# Reuse the same session setup as your fetcher
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
})

def polite_delay(min_s=1.5, max_s=3.5):
    time.sleep(random.uniform(min_s, max_s))

def fetch_results(year: str, output_file: str):
    # Load existing games to avoid duplicates
    existing = set()
    try:
        with open(output_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["date"], row["home_team"], row["away_team"]))
    except FileNotFoundError:
        pass

    new_rows = []

    for month in MONTHS:
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
        try:
            tqdm.write(f"Fetching {month.capitalize()} {year} results...")
            resp = session.get(url)
            resp.raise_for_status()
            polite_delay()
        except requests.HTTPError:
            tqdm.write(f"  ➜ {month} not available, skipping.")
            continue

        soup = bs4.BeautifulSoup(resp.content, "html.parser")
        tbody = soup.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        for tr in tqdm(rows, desc=f"Processing {month}", unit="game", leave=False):
            # Get date
            date_th = tr.find("th", {"data-stat": "date_game"})
            if not date_th:
                continue
            date_text = date_th.text.strip()
            try:
                game_date = datetime.datetime.strptime(date_text, "%a, %b %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                continue

            # Get teams
            home_td = tr.find("td", {"data-stat": "home_team_name"})
            away_td = tr.find("td", {"data-stat": "visitor_team_name"})
            if not home_td or not away_td:
                continue
            home_team = home_td.text.strip()
            away_team = away_td.text.strip()

            # Skip if already in file
            if (game_date, home_team, away_team) in existing:
                continue

            # Get scores (will be empty if game not yet played)
            home_pts_td = tr.find("td", {"data-stat": "home_pts"})
            away_pts_td = tr.find("td", {"data-stat": "visitor_pts"})
            if not home_pts_td or not away_pts_td:
                continue
            home_pts_text = home_pts_td.text.strip()
            away_pts_text = away_pts_td.text.strip()
            if not home_pts_text or not away_pts_text:
                continue  # game not yet played / no result

            try:
                home_pts = int(home_pts_text)
                away_pts = int(away_pts_text)
            except ValueError:
                continue

            # winning_team: 0 = home, 1 = away
            winning_team = 0 if home_pts > away_pts else 1

            new_rows.append([game_date, home_team, away_team, winning_team])
            existing.add((game_date, home_team, away_team))

    if not new_rows:
        tqdm.write("No new results found.")
        return

    # Append to CSV
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header only if file was empty before
        if not existing:
            writer.writerow(["date", "home_team", "away_team", "winning_team"])
        writer.writerows(new_rows)

    tqdm.write(f"✅ Appended {len(new_rows)} new game results to {output_file}")

if __name__ == "__main__":
    fetch_results(YEAR, OUTPUT_FILE)
    print("\nDone!\n")


# **AVERAGER**

# In[4]:


"""
averager.py – Computes rolling averages, Elo ratings, rest features,
and home/away performance splits for the DeepShot training pipeline.

Phase 1 Upgrades
----------------
1.1  Rest Advantage : days_since_last_game, is_back_to_back
1.2  Home/Away Splits : per-location rolling averages for key stats
1.3  Leakage Prevention : all rolling computations use .shift(1);
     rest/location metadata never included in rolled stat_columns.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EloConfig:
    """Configuration for Elo calculation."""
    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 65.0
    carry_over: float = 0.75
    apply_mov_multiplier: bool = True
    min_margin: float = 1.0
    season_start_month: int = 10  # October for NBA


game_window: int = 10

# Stats for which separate home/away rolling averages are computed (Phase 1.2)
SPLIT_STATS: List[str] = ["pts", "fg_pct", "fg3_pct", "ortg", "drtg"]

# Columns that must NEVER be rolled (pre-game metadata or post-game outcomes)
META_COLS: frozenset = frozenset(
    ["date", "team", "elo", "location", "won",
     "days_since_last_game", "is_back_to_back"]
)


# ── Elo Helpers ───────────────────────────────────────────────────────────────

def season_key(game_date: pd.Timestamp, season_start_month: int) -> int:
    """Return a season identifier based on the month a season starts."""
    return (
        game_date.year
        if game_date.month >= season_start_month
        else game_date.year - 1
    )


def load_results(results_file: Optional[str]) -> Optional[pd.DataFrame]:
    """Load results CSV if available."""
    if results_file and os.path.exists(results_file):
        results = pd.read_csv(results_file, parse_dates=["date"])
        if "winning_team" in results.columns:
            results["winning_team"] = results["winning_team"].astype(int)
        return results
    return None


def lookup_result(
    results_df: Optional[pd.DataFrame],
    game_date: pd.Timestamp,
    home_team: str,
    away_team: str,
) -> Optional[int]:
    """Return 1 if home won, 0 if away won, or None if not found."""
    if results_df is None or "winning_team" not in results_df.columns:
        return None
    mask = (
        (results_df["date"] == game_date)
        & (results_df["home_team"] == home_team)
        & (results_df["away_team"] == away_team)
    )
    if mask.any():
        return 1 - int(results_df.loc[mask, "winning_team"].iloc[0])
    return None


def compute_team_elos(
    gamelogs_df: pd.DataFrame,
    schedule_file: str,
    results_file: Optional[str] = None,
    config: Optional[EloConfig] = None,
) -> pd.DataFrame:
    """Compute pre-game Elo ratings per team for each played game."""
    cfg = config or EloConfig()

    games = pd.read_csv(schedule_file, parse_dates=["date"])
    games = games.sort_values("date")

    results_df = load_results(results_file)

    pts_lookup: Dict[Tuple[pd.Timestamp, str], float] = (
        gamelogs_df.assign(date=pd.to_datetime(gamelogs_df["date"]))
        .set_index(["date", "team"])["pts"]
        .astype(float)
        .to_dict()
    )

    ratings: Dict[str, float] = {}
    records: List[Dict[str, object]] = []
    current_season: Optional[int] = None

    for _, game in games.iterrows():
        game_date: pd.Timestamp = game["date"]
        season = season_key(game_date, cfg.season_start_month)

        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = {
                team: cfg.carry_over * rating
                + (1 - cfg.carry_over) * cfg.base_rating
                for team, rating in ratings.items()
            }
            current_season = season

        home_team: str = game["home_team"]
        away_team: str = game["away_team"]

        home_rating: float = ratings.get(home_team, cfg.base_rating)
        away_rating: float = ratings.get(away_team, cfg.base_rating)

        home_pts = pts_lookup.get((game_date, home_team))
        away_pts = pts_lookup.get((game_date, away_team))

        margin: Optional[float] = None
        if home_pts is not None and away_pts is not None:
            margin = max(abs(float(home_pts) - float(away_pts)), cfg.min_margin)

        actual_home: Optional[float] = lookup_result(
            results_df, game_date, home_team, away_team
        )
        if actual_home is None and home_pts is not None and away_pts is not None:
            actual_home = 1.0 if float(home_pts) > float(away_pts) else 0.0

        if actual_home is None:
            continue

        expected_home = 1 / (
            1
            + math.pow(
                10,
                ((away_rating - (home_rating + cfg.home_advantage)) / 400),
            )
        )

        mov_multiplier: float = 1.0
        if cfg.apply_mov_multiplier and margin is not None:
            mov_multiplier = math.log(margin + 1) * (
                2.2 / (abs(home_rating - away_rating) * 0.001 + 2.2)
            )

        delta = cfg.k_factor * mov_multiplier * (actual_home - expected_home)

        # Record PRE-GAME ratings (leakage-free)
        records.append({"date": game_date, "team": home_team, "elo": round(home_rating, 2)})
        records.append({"date": game_date, "team": away_team, "elo": round(away_rating, 2)})

        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    return pd.DataFrame(records)


# ── Phase 1.1 & 1.2 Helpers ───────────────────────────────────────────────────

def _build_location_map(schedule_file: str) -> Dict[Tuple[pd.Timestamp, str], str]:
    """Build a (date, team) → 'home'/'away' lookup from the schedule CSV."""
    schedule = pd.read_csv(schedule_file, parse_dates=["date"])
    loc_map: Dict[Tuple[pd.Timestamp, str], str] = {}
    for _, row in schedule.iterrows():
        loc_map[(row["date"], row["home_team"])] = "home"
        loc_map[(row["date"], row["away_team"])] = "away"
    return loc_map


def _build_win_map(results_file: Optional[str]) -> Dict[Tuple[pd.Timestamp, str], int]:
    """Build a (date, team) → 1/0 (won/lost) lookup from the results CSV."""
    if not results_file or not os.path.exists(results_file):
        return {}
    results = pd.read_csv(results_file, parse_dates=["date"])
    if "winning_team" not in results.columns:
        return {}
    results["winning_team"] = results["winning_team"].astype(int)
    win_map: Dict[Tuple[pd.Timestamp, str], int] = {}
    for _, row in results.iterrows():
        home_won = int(row["winning_team"]) == 1
        win_map[(row["date"], row["home_team"])] = 1 if home_won else 0
        win_map[(row["date"], row["away_team"])] = 0 if home_won else 1
    return win_map


def compute_location_split_rolling(
    df: pd.DataFrame,
    stat_cols: List[str],
    window: int,
) -> pd.DataFrame:
    """Compute separate rolling averages for home-only and away-only games."""
    df = df.copy()
    new_cols = [
        f"{col}_{loc}_avg"
        for loc in ("home", "away")
        for col in stat_cols
    ]
    for col in new_cols:
        df[col] = np.nan

    for team, group in df.groupby("team", observed=True):
        group = group.sort_values("date").copy()
        group_indices = group.index.tolist()

        for loc in ("home", "away"):
            loc_mask = group["location"] == loc
            loc_games = group[loc_mask]

            if loc_games.empty:
                continue

            loc_indices = loc_games.index.tolist()

            for col in stat_cols:
                if col not in loc_games.columns:
                    continue

                vals = loc_games[col].astype(float)
                # Phase 1.3: shift(1) → only past games are visible
                rolled = vals.rolling(window=window, min_periods=1).mean().shift(1)

                # Handle first value: use actual observed value if no prior data
                if len(rolled) > 0 and rolled.iloc[0] != rolled.iloc[0]:  # NaN check
                    rolled.iloc[0] = vals.iloc[0] if len(vals) > 0 else np.nan

                df.loc[loc_indices, f"{col}_{loc}_avg"] = rolled.values

        # Forward-fill and back-fill within team to propagate split values
        for col in new_cols:
            df.loc[group_indices, col] = df.loc[group_indices, col].ffill().bfill()

    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def compute_rolling_averages(
    game_window: int,
    gamelogs_file: str,
    output_file: str,
    schedule_file: str = "./csv/schedule.csv",
    results_file: str = "./csv/results.csv",
    elo_config: Optional[EloConfig] = None,
) -> None:
    """Compute rolling averages, rest features, home/away splits, and Elo ratings."""
    if game_window < 1:
        raise ValueError(f"game_window must be >= 1, got {game_window}")

    cfg = elo_config or EloConfig()

    # ── 1. Load & Sort ────────────────────────────────────────────────────────
    tqdm.write("Loading CSV file...")
    df: pd.DataFrame = pd.read_csv(gamelogs_file)
    tqdm.write(f"   Loaded {len(df)} game records")

    tqdm.write("   Sorting data by team and date...")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # ── 2. Base Feature Engineering ───────────────────────────────────────────
    tqdm.write("   Engineering additional features...")
    df["ast_tov"] = (df["ast"] / df["tov"].replace(0, np.nan)).round(2)
    df["ast_ratio"] = (
        df["ast"] / (df["fg"] + df["ast"] + df["tov"].replace(0, np.nan))
    ).round(2)

    # Preserve raw points for Elo (before smoothing)
    elo_input = df[["date", "team", "pts"]].copy()

    # ── 3. Phase 1.1: Rest Advantage Features ─────────────────────────────────
    tqdm.write("   Computing rest advantage features (Phase 1.1)...")
    df["days_since_last_game"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .fillna(7.0)   # First game of dataset: assume 7 days rest
    )
    df["is_back_to_back"] = (df["days_since_last_game"] < 2).astype(int)

    # ── 4. Phase 1.2: Location & Win Tagging ─────────────────────────────────
    tqdm.write("   Tagging home/away locations (Phase 1.2)...")
    loc_map = _build_location_map(schedule_file)
    df["location"] = df.apply(
        lambda r: loc_map.get((r["date"], r["team"]), "home"), axis=1
    )

    win_map = _build_win_map(results_file)
    df["won"] = df.apply(
        lambda r: float(win_map.get((r["date"], r["team"]), np.nan)), axis=1
    )

    # ── 5. Phase 1.3: Rolling Averages with .shift(1) ─────────────────────────
    stat_columns: List[str] = [c for c in df.columns if c not in META_COLS]
    tqdm.write(f"   Processing {len(stat_columns)} statistical columns")

    teams = df["team"].unique()
    tqdm.write(f"\nComputing rolling averages for {len(teams)} teams...\n")

    def compute_combined_avg(group: pd.DataFrame) -> pd.DataFrame:
        """Combined rolling mean + EWMA, shifted by 1 game (Phase 1.3)."""
        if len(group) == 0:
            return group[stat_columns]

        # Simple rolling mean
        rolling_mean = (
            group[stat_columns]
            .rolling(window=game_window, min_periods=1)
            .mean()
            .shift(1)
        )
        # EWMA – emphasises recent games
        ewma = (
            group[stat_columns]
            .ewm(span=game_window, adjust=False)
            .mean()
            .shift(1)
        )
        combined = 0.43 * rolling_mean + 0.57 * ewma

        # Seed first row with actual values (avoids NaN after shift)
        if len(combined) > 0:
            combined.iloc[0] = group.iloc[0][stat_columns]
        return combined

    tqdm.write("Computing averages per team...")
    # Snapshot of original per-game stats BEFORE smoothing (used for splits below)
    original_stats = df[["date", "team", "location", "won"] + stat_columns].copy()

    df[stat_columns] = df.groupby("team", group_keys=False, observed=True)[
        stat_columns
    ].apply(compute_combined_avg)

    tqdm.write("\nRounding values...")
    df[stat_columns] = df[stat_columns].round(2)

    # ── 6. Phase 1.2: Home/Away Split Rolling Averages ────────────────────────
    tqdm.write("\nComputing home/away split rolling averages (Phase 1.2)...")
    split_stats = [s for s in SPLIT_STATS if s in original_stats.columns] + ["won"]
    split_df = compute_location_split_rolling(original_stats, split_stats, game_window)

    split_only_cols = [
        c for c in split_df.columns
        if c.endswith("_home_avg") or c.endswith("_away_avg")
    ]
    if split_only_cols:
        df = df.merge(
            split_df[["date", "team"] + split_only_cols],
            on=["date", "team"],
            how="left",
        )
        df[split_only_cols] = df[split_only_cols].round(4)

    # ── 7. Elo Ratings ────────────────────────────────────────────────────────
    tqdm.write("\nComputing Elo ratings...")
    elo_history = compute_team_elos(
        gamelogs_df=elo_input,
        schedule_file=schedule_file,
        results_file=results_file,
        config=cfg,
    )
    df = df.merge(elo_history, on=["date", "team"], how="left")
    df["elo"] = (
        df.groupby("team")["elo"]
        .ffill()
        .fillna(cfg.base_rating)
        .round(2)
    )

    # ── 8. Save ───────────────────────────────────────────────────────────────
    if os.path.exists(output_file):
        tqdm.write(f"File {output_file} already exists. Removing...")
        os.remove(output_file)

    tqdm.write(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    tqdm.write("Rolling averages saved successfully!")
    tqdm.write(f"Output: {output_file}")
    tqdm.write(
        f"\nNew columns added:\n"
        f"  • days_since_last_game, is_back_to_back  (Phase 1.1)\n"
        f"  • {', '.join(split_only_cols[:4])} ...  (Phase 1.2)\n"
        f"  • location, won  (internal metadata)\n"
    )


if __name__ == "__main__":
    print(f"Game window size: {game_window}\n")
    compute_rolling_averages(game_window, "./csv/gamelogs.csv", "./csv/averages.csv")
    print("\nDone!\n")

