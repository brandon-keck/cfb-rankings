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

# Be polite to the free API tier -- small pause between requests
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

    # 2. Teams: names, conferences, IDs -- needed for strength-of-schedule grouping
    print("  Fetching teams...")
    teams = fetch("/teams/fbs", {"year": year}, api_key)
    save_json_as_csv(teams, OUTPUT_DIR / f"teams_{year}.csv")

    # 3. Talent: 247-style recruiting/roster talent composite per team.
    #    This is what seeds a preseason prior rating before any games
    #    are played, so week-1 rankings aren't just noise.
    print("  Fetching talent composite...")
    talent = fetch("/talent", {"year": year}, api_key)
    save_json_as_csv(talent, OUTPUT_DIR / f"talent_{year}.csv")

    # 4. SP+ ratings: useful later to sanity-check your own rankings against
    #    an established composite rating
    print("  Fetching SP+ ratings...")
    sp_ratings = fetch("/ratings/sp", {"year": year}, api_key)
    save_json_as_csv(sp_ratings, OUTPUT_DIR / f"sp_ratings_{year}.csv")
    # Combined *_all_years.csv files are rebuilt once, after all years are
    # fetched, from every per-year file on disk (see rebuild_combined_files) --
    # not appended here, so a narrow-range run can never wipe other years out.



def rebuild_combined_files() -> None:
    """Rebuilds each *_all_years.csv from EVERY per-year file currently on
    disk, regardless of which years this particular run fetched. This is
    what keeps the combined files trustworthy: without it, running this
    script for just one year (as the weekly automation does) would wipe
    every other year out of the combined files, even though the per-year
    files themselves are untouched."""
    file_groups = {
        "games_all_years.csv": "games_*.csv",
        "teams_all_years.csv": "teams_*.csv",
        "talent_all_years.csv": "talent_*.csv",
        "sp_ratings_all_years.csv": "sp_ratings_*.csv",
    }
    for combined_name, pattern in file_groups.items():
        per_year_files = sorted(
            p for p in OUTPUT_DIR.glob(pattern) if p.name != combined_name
        )
        if not per_year_files:
            continue

        all_rows = []
        fieldnames = []
        for path in per_year_files:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
                for k in reader.fieldnames or []:
                    if k not in fieldnames:
                        fieldnames.append(k)

        combined_path = OUTPUT_DIR / combined_name
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  rebuilt {combined_name} from {len(per_year_files)} per-year file(s), {len(all_rows)} total rows")


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

    print(f"Fetching {args.start_year}-{args.end_year} {args.season_type} season data from CFBD...")

    for year in range(args.start_year, args.end_year + 1):
        fetch_year(year, args.season_type, api_key)

    print("\nRebuilding combined multi-year files from all per-year files on disk...")
    rebuild_combined_files()

    print("\nDone. Per-year and combined multi-year files are in the 'data/' folder.")


if __name__ == "__main__":
    main()