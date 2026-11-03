import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rank_teams import DATA_DIR, load_games, load_talent, load_fbs_teams, run_elo

HOME_ADV_SP_PLUS = 2.5  # a modest, standard home-field point edge for the SP+ comparison


def load_sp_plus() -> dict[tuple[int, str], float]:
    """Returns {(season, team): sp_plus_overall_rating}, built from
    per-year sp_ratings_YYYY.csv files directly (not the combined file),
    for the same robustness reason as rank_teams.py's load_talent()."""
    import glob
    import re

    sp = {}
    for path in sorted(glob.glob(str(DATA_DIR / "sp_ratings_*.csv"))):
        p = Path(path)
        if p.name == "sp_ratings_all_years.csv":
            continue
        m = re.match(r"sp_ratings_(\d{4})\.csv$", p.name)
        if not m:
            continue
        season = int(m.group(1))
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    team = (row.get("team") or "").strip()
                    rating = float(row.get("rating"))
                    sp[(season, team)] = rating
                except (TypeError, ValueError):
                    continue
    return sp


def brier_score(predictions: list[dict]) -> float:
    total = 0.0
    for p in predictions:
        actual = 1.0 if p["home_won"] else 0.0
        total += (p["home_win_prob"] - actual) ** 2
    return total / len(predictions) if predictions else float("nan")


def accuracy(predictions: list[dict], prob_key: str = "home_win_prob") -> float:
    correct = 0
    for p in predictions:
        predicted_home_win = p[prob_key] > 0.5
        if predicted_home_win == p["home_won"]:
            correct += 1
    return correct / len(predictions) if predictions else float("nan")


CALIBRATION_BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]


def was_correct(p: dict) -> bool:
    """Same rule accuracy() uses: the model 'picked' the home team if it gave
    them better than a 50% chance."""
    return (p["home_win_prob"] > 0.5) == p["home_won"]


def calibration_table(predictions: list[dict]) -> list[dict]:
    """Group games by how confident the model was in its favored team, then
    compare that confidence to how often the favorite actually won. A
    well-calibrated model's 70% favorites win about 70% of the time."""
    table = []
    for lo, hi in CALIBRATION_BINS:
        bucket = []
        for p in predictions:
            p_home = p["home_win_prob"]
            fav_prob = max(p_home, 1 - p_home)
            if lo <= fav_prob < hi:
                fav_won = p["home_won"] if p_home >= 0.5 else (not p["home_won"])
                bucket.append((fav_prob, fav_won))
        if not bucket:
            continue
        table.append(
            {
                "bucket": f"{round(lo * 100)}-{min(round(hi * 100), 100)}%",
                "games": len(bucket),
                "avg_predicted": round(sum(b[0] for b in bucket) / len(bucket), 4),
                "actual_win_rate": round(sum(1 for b in bucket if b[1]) / len(bucket), 4),
            }
        )
    return table


def recent_slate(predictions: list[dict], days: int = 6) -> list[dict]:
    """The most recent stretch of completed games (anchored on the latest
    completed game's date, not today's date, so it behaves the same no matter
    when the script runs). Used for the 'last week's picks, graded' list."""
    dated = [p for p in predictions if p.get("game_date")]
    if not dated:
        return []
    latest = datetime.strptime(max(p["game_date"] for p in dated), "%Y-%m-%d")
    cutoff = (latest - timedelta(days=days)).strftime("%Y-%m-%d")
    slate = [p for p in dated if p["game_date"] >= cutoff]
    return sorted(slate, key=lambda p: (p["game_date"], p["home_team"]))


def main():
    print("Loading games, talent, and FBS rosters (same as rank_teams.py)...")
    games = load_games()
    talent_by_team = load_talent()
    fbs_teams = load_fbs_teams()

    print("Running Elo model and capturing pre-game predictions...")
    _weekly_rows, _season_final_rows, predictions, _full_field_weekly_rows = run_elo(games, talent_by_team, fbs_teams)

    if not predictions:
        print("No FBS-vs-FBS games found to backtest. Run cfb.py and rank_teams.py first.")
        return

    print(f"\n{len(predictions)} FBS-vs-FBS games evaluated.\n")

    # ---- Elo model performance ----
    elo_acc = accuracy(predictions)
    elo_brier = brier_score(predictions)
    print(f"Elo model accuracy:        {elo_acc:.1%}")
    print(f"Elo model Brier score:     {elo_brier:.4f}  (0 = perfect, 0.25 = coin-flip guessing)")

    # ---- Baseline: home team always wins ----
    home_baseline_preds = [{"home_win_prob": 1.0, "home_won": p["home_won"]} for p in predictions]
    home_acc = accuracy(home_baseline_preds)
    print(f"\nBaseline (home always wins) accuracy: {home_acc:.1%}")
    print(f"  Elo model {'beats' if elo_acc > home_acc else 'does not beat'} this naive baseline by {abs(elo_acc - home_acc):.1%}")

    # ---- Baseline: SP+ rating differential ----
    sp_plus = load_sp_plus()
    sp_predictions = []
    for p in predictions:
        home_sp = sp_plus.get((p["season"], p["home_team"]))
        away_sp = sp_plus.get((p["season"], p["away_team"]))
        if home_sp is None or away_sp is None:
            continue
        point_diff = (home_sp - away_sp) + HOME_ADV_SP_PLUS
        # Convert a point-spread-like differential to a probability using
        # the same logistic shape as the Elo model, just rescaled.
        prob = 1.0 / (1.0 + 10 ** (-point_diff / 14.0))
        sp_predictions.append({"home_win_prob": prob, "home_won": p["home_won"]})

    if sp_predictions:
        sp_acc = accuracy(sp_predictions)
        sp_brier = brier_score(sp_predictions)
        print(f"\nSP+ baseline accuracy:     {sp_acc:.1%}  (on the {len(sp_predictions)} games where SP+ data exists)")
        print(f"SP+ baseline Brier score:  {sp_brier:.4f}")
        print(f"  Elo model {'beats' if elo_acc > sp_acc else 'does not beat'} SP+ on accuracy by {abs(elo_acc - sp_acc):.1%}")
        print("  Note: SP+ uses full-season data, so this isn't a strictly fair no-look-ahead comparison.")
    else:
        print("\nNo overlapping SP+ data found to compare against.")

    # ---- Per-season breakdown ----
    seasons = sorted(set(p["season"] for p in predictions))
    per_season = []
    print("\nAccuracy by season:")
    for season in seasons:
        season_preds = [p for p in predictions if p["season"] == season]
        if len(season_preds) < 10:
            continue  # skip seasons with too few completed games to be meaningful (e.g. current in-progress season early on)
        season_acc = accuracy(season_preds)
        per_season.append({"season": season, "accuracy": round(season_acc, 4), "games": len(season_preds)})
        print(f"  {season}: {season_acc:.1%}  ({len(season_preds)} games)")

    # ---- Save full results for further analysis ----
    out_path = DATA_DIR / "backtest_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["season", "week", "game_date", "kickoff_utc", "home_team", "away_team", "home_win_prob", "home_won", "home_points", "away_points", "predicted_correctly"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            row = dict(p)
            row["predicted_correctly"] = was_correct(p)
            writer.writerow(row)
    print(f"\nSaved per-game predictions -> {out_path}")

    # ---- Calibration: does "70%" actually mean 70%? ----
    calibration = calibration_table(predictions)
    print("\nCalibration (when the model favors a team at X%, how often does that team win?):")
    for row in calibration:
        print(f"  {row['bucket']:>8}: {row['games']:>5} games, predicted {row['avg_predicted']:.1%}, actually won {row['actual_win_rate']:.1%}")

    # ---- Current season record so far ----
    latest_season = max(p["season"] for p in predictions)
    current = [p for p in predictions if p["season"] == latest_season]
    current_correct = sum(1 for p in current if was_correct(p))
    print(f"\n{latest_season} so far: {current_correct} of {len(current)} games called correctly ({current_correct / len(current):.1%})")

    # ---- Most recent slate of games, graded ----
    slate = recent_slate(predictions)
    recent_path = DATA_DIR / "recent_results.csv"
    with open(recent_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["season", "week", "game_date", "kickoff_utc", "home_team", "away_team", "home_win_prob", "home_points", "away_points", "home_won", "predicted_correctly"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in slate:
            row = dict(p)
            row["predicted_correctly"] = was_correct(p)
            writer.writerow(row)
    slate_correct = sum(1 for p in slate if was_correct(p))
    print(f"Most recent slate: {slate_correct} of {len(slate)} correct -> {recent_path}")

    # ---- Save a small summary JSON -- this is what the README and the
    # dashboard read from, so both stay current automatically instead of
    # having yesterday's numbers typed in by hand.
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "games_evaluated": len(predictions),
        "elo_accuracy": round(elo_acc, 4),
        "elo_brier_score": round(elo_brier, 4),
        "home_baseline_accuracy": round(home_acc, 4),
        "elo_vs_home_baseline_pts": round((elo_acc - home_acc) * 100, 1),
        "sp_plus_accuracy": round(sp_acc, 4) if sp_predictions else None,
        "sp_plus_brier_score": round(sp_brier, 4) if sp_predictions else None,
        "sp_plus_games_compared": len(sp_predictions) if sp_predictions else 0,
        "elo_vs_sp_plus_pts": round((elo_acc - sp_acc) * 100, 1) if sp_predictions else None,
        "accuracy_by_season": per_season,
        "current_season": {
            "season": latest_season,
            "correct": current_correct,
            "games": len(current),
            "accuracy": round(current_correct / len(current), 4),
        },
        "calibration": calibration,
    }
    summary_path = DATA_DIR / "backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()