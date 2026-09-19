"""
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
"""

import csv
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


def main():
    print("Loading games, talent, and FBS rosters (same as rank_teams.py)...")
    games = load_games()
    talent_by_team = load_talent()
    fbs_teams = load_fbs_teams()

    print("Running Elo model and capturing pre-game predictions...")
    _weekly_rows, _season_final_rows, predictions = run_elo(games, talent_by_team, fbs_teams)

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
    print("\nAccuracy by season:")
    for season in seasons:
        season_preds = [p for p in predictions if p["season"] == season]
        if len(season_preds) < 10:
            continue  # skip seasons with too few completed games to be meaningful (e.g. current in-progress season early on)
        print(f"  {season}: {accuracy(season_preds):.1%}  ({len(season_preds)} games)")

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


if __name__ == "__main__":
    main()