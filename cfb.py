"""
Pulls multiple seasons of college football games, teams, talent
(recruiting composite), and SP+ ratings from the CollegeFootballData
(CFBD) API and saves them to local CSV files. Built to support an
in-season, week-by-week ranking algorithm (Elo-style ratings seeded
with a preseason prior).

Setup:
    1. Get a free API key at https://collegefootballdata.com
    2. Set it as an environment variable before running this script:

       macOS/Linux:
           export CFBD_API_KEY="your_key_here"

       Windows (PowerShell):
           $env:CFBD_API_KEY="your_key_here"

    3. Run (defaults to 2019-2024):
           python fetch_cfbd_data.py
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.collegefootballdata.com"
OUTPUT_DIR = Path("data")


REQUEST_DELAY_SECONDS = 0.5


def get_api_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit(
            "ERROR: CFBD_API_KEY environment variable not set.\n"
            "Run: export CFBD_API_KEY='your_key_here'  (macOS/Linux)\n"
            "  or: $env:CFBD_API_KEY='your_key_here'    (Windows PowerShell)"
        )
    return key


def fetch(endpoint: str, params: dict, api_key: str) -> list:
    """GET a CFBD endpoint and return the parsed JSON (a list of records)."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response.json()


def save_json_as_csv(records: list, filepath: Path, mode: str = "w") -> None:
    """
    Save a list of flat dicts to CSV using only the standard library.
    mode="a" appends without rewriting the header (used to build
    combined multi-year files incrementally).
    """
    if not records:
        print(f"    (no records returned, skipping {filepath.name})")
        return

    # Union of all keys across records, in case some records have extra fields
    fieldnames = []
    for r in records:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    write_header = mode == "w" or not filepath.exists()

    with open(filepath, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(records)

    action = "saved" if write_header else "appended"
    print(f"    {action} {len(records)} rows -> {filepath}")


def fetch_year(year: int, season_type: str, api_key: str) -> None:
    print(f"\n=== {year} ===")

    # 1. Games: final scores, teams, dates -- the core table for a ranking algorithm
    print("  Fetching games...")
    games = fetch(
        "/games",
        {"year": year, "seasonType": season_type, "division": "fbs"},
        api_key,
    )
    save_json_as_csv(games, OUTPUT_DIR / f"games_{year}.csv")
    save_json_as_csv(games, OUTPUT_DIR / "games_all_years.csv", mode="a")

    # 2. Teams: names, conferences, IDs -- needed for strength-of-schedule grouping
    print("  Fetching teams...")
    teams = fetch("/teams/fbs", {"year": year}, api_key)
    save_json_as_csv(teams, OUTPUT_DIR / f"teams_{year}.csv")
    save_json_as_csv(teams, OUTPUT_DIR / "teams_all_years.csv", mode="a")

    # 3. Talent: 247-style recruiting/roster talent composite per team.
    #    This is what seeds a preseason prior rating before any games
    #    are played, so week-1 rankings aren't just noise.
    print("  Fetching talent composite...")
    talent = fetch("/talent", {"year": year}, api_key)
    save_json_as_csv(talent, OUTPUT_DIR / f"talent_{year}.csv")
    save_json_as_csv(talent, OUTPUT_DIR / "talent_all_years.csv", mode="a")

    # 4. SP+ ratings: useful later to sanity-check your own rankings against
    #    an established composite rating
    print("  Fetching SP+ ratings...")
    sp_ratings = fetch("/ratings/sp", {"year": year}, api_key)
    save_json_as_csv(sp_ratings, OUTPUT_DIR / f"sp_ratings_{year}.csv")
    save_json_as_csv(sp_ratings, OUTPUT_DIR / "sp_ratings_all_years.csv", mode="a")


def main():
    parser = argparse.ArgumentParser(
        description="Pull multiple seasons of CFB data from the CFBD API"
    )
    parser.add_argument("--start-year", type=int, default=2019, help="First season year to pull")
    parser.add_argument("--end-year", type=int, default=2024, help="Last season year to pull (inclusive)")
    parser.add_argument(
        "--season-type",
        default="regular",
        choices=["regular", "postseason"],
        help="Regular season or postseason games",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        sys.exit("ERROR: --start-year must be <= --end-year")

    api_key = get_api_key()

    # Clear old combined files so re-running the script doesn't duplicate rows
    for combined_name in ["games_all_years.csv", "teams_all_years.csv", "talent_all_years.csv", "sp_ratings_all_years.csv"]:
        combined_path = OUTPUT_DIR / combined_name
        if combined_path.exists():
            combined_path.unlink()

    print(f"Fetching {args.start_year}-{args.end_year} {args.season_type} season data from CFBD...")

    for year in range(args.start_year, args.end_year + 1):
        fetch_year(year, args.season_type, api_key)

    print("\nDone. Per-year and combined multi-year files are in the 'data/' folder.")


if __name__ == "__main__":
    main()