"""
backtest.py

Checks whether the Elo model's rankings actually predict outcomes well,
rather than just looking plausible. For every FBS-vs-FBS game, this uses
the exact same rating engine as rank_teams.py to see what the model would
have predicted BEFORE that game was played (no look-ahead), then compares
that prediction to what actually happened.

Metrics:
    - Accuracy: did the model favor the team that actually won?
    - Brier score: how well-calibrated were the win probabilities, not just
      the yes/no pick? Lower is better (0 = perfect, 0.25 = coin-flip
      guessing on every game, 1.0 = confidently wrong every time).
    - Compared against two baselines:
        1. "Home team always wins" -- tests whether the model earns its
           keep over the single most obvious naive heuristic in football.
        2. SP+ rating differential -- tests the model against an established
           external system. Note: SP+ ratings are computed using the whole
           season's data, so this comparison isn't perfectly apples-to-apples
           with the Elo model's no-look-ahead predictions -- it's a useful
           reference point, not a strict benchmark.

Run this after rank_teams.py has been run at least once (it reuses the
same rating engine and data-loading functions).

    python backtest.py

Writes:
    data/backtest_predictions.csv   every game's prediction vs. actual result
"""

import csv
import json
from datetime import datetime, timezone
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


def load_ap_poll() -> dict[tuple[int, int, str], int]:
    """Returns {(season, week, team): ap_rank} for the AP Top 25 poll only,
    built from per-year rankings_YYYY.csv files. NOTE: this assumes CFBD's
    "week 1" ranking is the preseason poll (see the comment in
    fetch_cfbd_data.py) so week N's poll is compared directly against week
    N's games with no offset."""
    import glob
    import re

    ap = {}
    for path in sorted(glob.glob(str(DATA_DIR / "rankings_*.csv"))):
        p = Path(path)
        if p.name == "rankings_all_years.csv":
            continue
        m = re.match(r"rankings_(\d{4})\.csv$", p.name)
        if not m:
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("poll") != "AP Top 25":
                    continue
                try:
                    season = int(row["season"])
                    week = int(row["week"])
                    team = (row.get("school") or "").strip()
                    rank = int(row["rank"])
                    ap[(season, week, team)] = rank
                except (KeyError, ValueError, TypeError):
                    continue
    return ap


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

    # ---- Baseline: AP Poll ----
    # Unlike Elo/SP+, the poll only gives a hard pick, not a calibrated
    # probability -- so only accuracy is reported for it, not a Brier score.
    # Games where NEITHER team is ranked are excluded entirely: the poll
    # offers no signal there, so including them would understate the poll's
    # real accuracy on the games it actually has an opinion about.
    ap_poll = load_ap_poll()
    ap_correct = 0
    ap_total = 0
    for p in predictions:
        home_rank = ap_poll.get((p["season"], p["week"], p["home_team"]))
        away_rank = ap_poll.get((p["season"], p["week"], p["away_team"]))
        if home_rank is None and away_rank is None:
            continue
        if home_rank is not None and away_rank is not None:
            predicted_home_win = home_rank < away_rank  # lower rank number = better
        else:
            predicted_home_win = home_rank is not None  # ranked team favored over unranked
        ap_total += 1
        if predicted_home_win == p["home_won"]:
            ap_correct += 1

    if ap_total:
        ap_acc = ap_correct / ap_total
        print(f"\nAP Poll baseline accuracy: {ap_acc:.1%}  (on the {ap_total} games where at least one team was ranked)")
        print(f"  Elo model {'beats' if elo_acc > ap_acc else 'does not beat'} the AP Poll on accuracy by {abs(elo_acc - ap_acc):.1%}")
        print("  Note: only a hard pick, not a calibrated probability, so no Brier score here.")
        print("  Note: assumes CFBD's week-1 poll is the preseason poll (see fetch_cfbd_data.py comment).")
    else:
        ap_acc = None
        print("\nNo overlapping AP Poll data found to compare against.")

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
        fieldnames = ["season", "week", "game_date", "home_team", "away_team", "home_win_prob", "home_won", "predicted_correctly"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            row = dict(p)
            row["predicted_correctly"] = (p["home_win_prob"] > 0.5) == p["home_won"]
            writer.writerow(row)
    print(f"\nSaved per-game predictions -> {out_path}")

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
        "ap_poll_accuracy": round(ap_acc, 4) if ap_total else None,
        "ap_poll_games_compared": ap_total,
        "elo_vs_ap_poll_pts": round((elo_acc - ap_acc) * 100, 1) if ap_total else None,
        "accuracy_by_season": per_season,
    }
    summary_path = DATA_DIR / "backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()