"""
How it works:
    - Every team starts a season with a rating carried over from the
      previous season, regressed partway back toward the average
      (1500) so last year's results don't dominate forever.
    - That carryover rating is blended with the team's recruiting
      talent composite for the new season, so roster turnover shows
      up in the preseason number instead of waiting for games to
      reveal it.
    - Teams with no rating history (first year in FBS, or first year
      in our data) start from a talent-only estimate, or 1500 if no
      talent data exists either.
    - After each completed game, both teams' ratings are updated
      based on the actual result vs. what the pre-game rating
      difference predicted, with a home-field-advantage bonus and a
      margin-of-victory multiplier (blowouts move ratings more than
      one-score games, but with diminishing returns).
    - A snapshot of every team's rating is saved after each week, so
      you can plot a team's rating trajectory across a season or
      reconstruct "what would the rankings have looked like in week 8."
"""

import csv
import glob
import math
import re
from pathlib import Path

DATA_DIR = Path("data")

DEFAULT_RATING = 1500.0
FCS_DEFAULT_RATING = 1150.0      # starting rating for non-FBS "buy game" opponents (FCS/D2/D3)
HOME_FIELD_ADVANTAGE = 65.0      # Elo points added to the home team's rating pre-game
BASE_K = 24.0                    # base update speed; higher = ratings move faster per game
REGRESSION_TO_MEAN = 0.65        # fraction of last season's rating strength kept into the next season
TALENT_BLEND_WEIGHT = 0.30       # weight given to talent-based prior vs. carryover rating


def find_season_game_files() -> list[Path]:
    """Prefer the combined file, but also pick up a current-season file
    (e.g. games_2026.csv) if it wasn't included in the combined pull."""
    files = []
    combined = DATA_DIR / "games_all_years.csv"
    if combined.exists():
        files.append(combined)

    seasons_in_combined = set()
    if combined.exists():
        with open(combined, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seasons_in_combined.add(row.get("season"))

    for path in sorted(glob.glob(str(DATA_DIR / "games_*.csv"))):
        p = Path(path)
        if p.name in ("games_all_years.csv",):
            continue
        m = re.match(r"games_(\d{4})\.csv$", p.name)
        if m and m.group(1) not in seasons_in_combined:
            files.append(p)

    return files


def load_games() -> list[dict]:
    games = []
    for path in find_season_game_files():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                games.append(row)

    parsed = []
    for g in games:
        completed = str(g.get("completed", "")).strip().lower() == "true"
        if not completed:
            continue
        try:
            season = int(g["season"])
            week = int(g["week"])
            home_points = int(g["homePoints"])
            away_points = int(g["awayPoints"])
        except (KeyError, ValueError, TypeError):
            continue  # skip malformed / incomplete rows

        parsed.append(
            {
                "season": season,
                "week": week,
                "home_team": g.get("homeTeam", "").strip(),
                "away_team": g.get("awayTeam", "").strip(),
                "home_points": home_points,
                "away_points": away_points,
                "neutral_site": str(g.get("neutralSite", "")).strip().lower() == "true",
                "start_date": g.get("startDate", ""),
            }
        )

    # Deduplicate in case the same game appears more than once (e.g. it
    # showed up in both a per-year file and the combined file, or CFBD
    # corrected a score between two pulls). Key on the matchup itself, not
    # the score, and keep the LAST-seen version -- since files are loaded
    # combined-file-first and per-year-fallback-second, and re-pulls always
    # contain the most current data, "last seen" is the freshest version.
    deduped_by_key = {}
    for g in parsed:
        key = (g["season"], g["week"], g["home_team"], g["away_team"])
        deduped_by_key[key] = g  # overwrite -- last occurrence wins

    deduped = list(deduped_by_key.values())
    deduped.sort(key=lambda g: (g["season"], g["week"], g["start_date"]))
    return deduped


def load_fbs_teams() -> dict[int, set[str]]:
    """Returns {season: {team names that were actually FBS that season}},
    built from the per-year teams_YYYY.csv files (pulled from CFBD's
    /teams/fbs endpoint, so these are the authoritative FBS rosters).
    Used to keep FCS/D2/D3 'buy game' opponents out of the rankings
    output, even though their games still count toward the FBS team's
    rating."""
    fbs_teams: dict[int, set[str]] = {}
    for path in sorted(glob.glob(str(DATA_DIR / "teams_*.csv"))):
        p = Path(path)
        if p.name == "teams_all_years.csv":
            continue
        m = re.match(r"teams_(\d{4})\.csv$", p.name)
        if not m:
            continue
        season = int(m.group(1))
        teams = set()
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("school") or row.get("team")
                if name:
                    teams.add(name.strip())
        fbs_teams[season] = teams
    return fbs_teams


def load_talent() -> dict[tuple[int, str], float]:
    """Returns {(season, team): talent_composite}"""
    talent = {}
    path = DATA_DIR / "talent_all_years.csv"
    if not path.exists():
        return talent

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                season = int(row["year"]) if "year" in row else int(row["season"])
                team = row.get("school") or row.get("team")
                score = float(row["talent"])
                talent[(season, team.strip())] = score
            except (KeyError, ValueError, TypeError):
                continue
    return talent


def talent_to_rating(talent_score: float, season_talent_values: list[float]) -> float:
    """Convert a raw talent composite into an Elo-like prior via z-score
    against that season's talent distribution."""
    if not season_talent_values or len(season_talent_values) < 2:
        return DEFAULT_RATING
    mean = sum(season_talent_values) / len(season_talent_values)
    variance = sum((v - mean) ** 2 for v in season_talent_values) / len(season_talent_values)
    stdev = math.sqrt(variance) if variance > 0 else 1.0
    z = (talent_score - mean) / stdev
    return DEFAULT_RATING + z * 100.0  # 1 std dev of talent ~ 100 Elo points


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(point_margin: int, elo_diff: float) -> float:
    """Margin-of-victory multiplier (adapted from FiveThirtyEight's NFL Elo).
    Blowouts move ratings more, but a huge favorite winning huge moves
    ratings less than an underdog winning huge."""
    return math.log(abs(point_margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))


def k_factor(season_game_number: int) -> float:
    """Slightly higher K early in the season while ratings are still
    settling, tapering to the base value."""
    if season_game_number <= 2:
        return BASE_K * 1.5
    if season_game_number <= 4:
        return BASE_K * 1.2
    return BASE_K


def seed_season_ratings(
    season: int,
    previous_final_ratings: dict[str, float],
    talent_by_team: dict[tuple[int, str], float],
):
    season_talent_values = [v for (s, _), v in talent_by_team.items() if s == season]

    def prior_for(team: str) -> float:
        carryover = previous_final_ratings.get(team)
        talent_score = talent_by_team.get((season, team))
        talent_rating = talent_to_rating(talent_score, season_talent_values) if talent_score is not None else None

        if carryover is not None:
            regressed = DEFAULT_RATING + (carryover - DEFAULT_RATING) * REGRESSION_TO_MEAN
            if talent_rating is not None:
                return regressed * (1 - TALENT_BLEND_WEIGHT) + talent_rating * TALENT_BLEND_WEIGHT
            return regressed

        if talent_rating is not None:
            return talent_rating

        return DEFAULT_RATING

    return prior_for


def run_elo(games: list[dict], talent_by_team: dict[tuple[int, str], float], fbs_teams: dict[int, set[str]]):
    ratings: dict[str, float] = {}
    games_played_this_season: dict[str, int] = {}
    weekly_rows = []
    season_final_rows = []

    current_season = None
    current_season_fbs: set[str] = set()

    def is_fbs(team: str, season: int) -> bool:
        known = fbs_teams.get(season)
        return team in known if known else True  # if we have no roster data, don't exclude anyone

    for g in games:
        if g["season"] != current_season:
            # New season: seed ratings using last season's finals + talent blend.
            # Only carry forward + rank teams that were actually FBS that season;
            # non-FBS "buy game" opponents are handled separately below.
            if current_season is not None:
                for team, rating in ratings.items():
                    if team in current_season_fbs:
                        season_final_rows.append({"season": current_season, "team": team, "final_rating": round(rating, 1)})

            prior_final_ratings = dict(ratings)
            prior_for = seed_season_ratings(g["season"], prior_final_ratings, talent_by_team)

            current_season_fbs = fbs_teams.get(g["season"])
            if not current_season_fbs:
                # No roster file for this season -- fall back to inferring
                # from game participants (better than nothing)
                current_season_fbs = {gm["home_team"] for gm in games if gm["season"] == g["season"]} | {
                    gm["away_team"] for gm in games if gm["season"] == g["season"]
                }

            ratings = {team: prior_for(team) for team in current_season_fbs}
            games_played_this_season = {team: 0 for team in current_season_fbs}

            current_season = g["season"]

        home, away = g["home_team"], g["away_team"]
        if home not in ratings:
            ratings[home] = DEFAULT_RATING if is_fbs(home, current_season) else FCS_DEFAULT_RATING
            games_played_this_season[home] = 0
        if away not in ratings:
            ratings[away] = DEFAULT_RATING if is_fbs(away, current_season) else FCS_DEFAULT_RATING
            games_played_this_season[away] = 0

        home_adv = 0.0 if g["neutral_site"] else HOME_FIELD_ADVANTAGE
        home_rating_with_field = ratings[home] + home_adv

        expected_home = expected_score(home_rating_with_field, ratings[away])
        actual_home = 1.0 if g["home_points"] > g["away_points"] else (0.0 if g["home_points"] < g["away_points"] else 0.5)

        margin = g["home_points"] - g["away_points"]
        elo_diff = home_rating_with_field - ratings[away]
        multiplier = mov_multiplier(margin, elo_diff)

        k = k_factor(min(games_played_this_season[home], games_played_this_season[away]))
        change = k * multiplier * (actual_home - expected_home)

        ratings[home] += change
        ratings[away] -= change
        games_played_this_season[home] += 1
        games_played_this_season[away] += 1

        # Snapshot both teams involved for this week (avoids writing every
        # team every week even on their bye weeks; still gives a full
        # picture once you pivot the data). Non-FBS buy-game opponents are
        # excluded from the output -- their game still updated the FBS
        # team's rating above, they just don't get ranked themselves.
        # game_date is included because a team can play more than once in
        # the same numbered week (early-season scheduling quirks), so week
        # number alone isn't a unique x-axis key for charting.
        game_date = g["start_date"][:10] if g["start_date"] else ""
        for team, opponent, team_points, opp_points in (
            (home, away, g["home_points"], g["away_points"]),
            (away, home, g["away_points"], g["home_points"]),
        ):
            if team in current_season_fbs:
                result = "W" if team_points > opp_points else ("L" if team_points < opp_points else "T")
                weekly_rows.append(
                    {
                        "season": g["season"],
                        "week": g["week"],
                        "game_date": game_date,
                        "team": team,
                        "opponent": opponent,
                        "result": f"{result} {team_points}-{opp_points}",
                        "rating": round(ratings[team], 1),
                    }
                )

    if current_season is not None:
        for team, rating in ratings.items():
            if team in current_season_fbs:
                season_final_rows.append({"season": current_season, "team": team, "final_rating": round(rating, 1)})

    return weekly_rows, season_final_rows


def add_ranks_within_group(rows: list[dict], group_keys: list[str], value_key: str) -> None:
    from itertools import groupby

    rows.sort(key=lambda r: tuple(r[k] for k in group_keys) + (-r[value_key],))
    for _, group in groupby(rows, key=lambda r: tuple(r[k] for k in group_keys)):
        for i, row in enumerate(group, start=1):
            row["rank"] = i


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {len(rows)} rows -> {path}")


def main():
    print("Loading games...")
    games = load_games()
    if not games:
        print("No completed games found in data/. Run fetch_cfbd_data.py first.")
        return
    print(f"  {len(games)} completed games loaded, seasons {games[0]['season']}-{games[-1]['season']}")

    print("Loading talent composite...")
    talent_by_team = load_talent()
    print(f"  {len(talent_by_team)} team-season talent records loaded")

    print("Loading FBS team rosters (to exclude buy-game opponents from rankings)...")
    fbs_teams = load_fbs_teams()
    print(f"  FBS rosters loaded for seasons: {sorted(fbs_teams.keys())}")

    print("Running Elo model...")
    weekly_rows, season_final_rows = run_elo(games, talent_by_team, fbs_teams)

    add_ranks_within_group(weekly_rows, ["season", "week"], "rating")
    add_ranks_within_group(season_final_rows, ["season"], "final_rating")

    save_csv(weekly_rows, DATA_DIR / "elo_ratings_weekly.csv", ["season", "week", "game_date", "team", "opponent", "result", "rating", "rank"])
    save_csv(season_final_rows, DATA_DIR / "elo_ratings_season_final.csv", ["season", "team", "final_rating", "rank"])

    latest_season = max(r["season"] for r in season_final_rows)
    print(f"\nCurrent Top 25 standings, season {latest_season} (through the most recently played games):")
    top25 = sorted(
        (r for r in season_final_rows if r["season"] == latest_season),
        key=lambda r: r["rank"],
    )[:25]
    for r in top25:
        print(f"  {r['rank']:>2}. {r['team']:<25} {r['final_rating']}")


if __name__ == "__main__":
    main()