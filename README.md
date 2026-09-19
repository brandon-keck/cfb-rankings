# CFB Rankings

In-season college football power rankings built on an Elo-style rating model,
seeded each year with a blend of prior-season performance and recruiting
talent data. Rankings update automatically as new games are played, instead
of relying on human polls.

## Why

The AP Poll and Coaches Poll are voted on by humans, which means they carry
the same biases every subjective ranking does: preseason reputation, name
recognition, and week-to-week overreaction. This project is an attempt at a
transparent, fully reproducible alternative — every rating is the output of
a documented formula applied to real game results, not a ballot.

## How the rankings work

Each team's rating starts the season as a blend of two things:

1. **Carryover rating** — last season's final rating, regressed 65% of the
   way back toward the average (1500). This keeps last year's champion from
   starting the new season with an insurmountable head start; they have to
   keep proving it.
2. **Recruiting talent composite** — blended in at 30% weight, so roster
   turnover (a strong recruiting class, a mass exodus to the transfer
   portal) shows up in the preseason number instead of only being revealed
   game by game.

From there, every completed game updates both teams' ratings based on the
**actual result vs. what the pre-game rating difference predicted** — the
core idea behind Elo ratings, originally developed for chess. A few
refinements on top of vanilla Elo:

- **Home-field advantage**: home teams get a +65 rating bonus when
  calculating the expected outcome (no bonus on neutral-site games).
- **Margin-of-victory multiplier**: blowouts move ratings more than
  one-score wins, but with diminishing returns — a heavy favorite winning
  big barely moves the needle, since that outcome was already expected.
- **Early-season volatility**: each team's first couple of games in a
  season use a higher K-factor (the "how much should this game move the
  rating" constant), since early results are noisier and ratings should
  adapt faster before settling down.
- **Non-FBS opponents**: FBS teams occasionally play an FCS/Division
  II/III "buy game." Those games still count toward the FBS team's rating,
  but the non-FBS opponent itself is never included in the rankings output.

## Data source

All data comes from the free tier of the
[CollegeFootballData.com](https://collegefootballdata.com) (CFBD) API —
games, team rosters, recruiting talent composite scores, and SP+ ratings
(used as an external sanity check, not as an input to the model itself).

## Project structure

```
cfb-rankings/
├── cfb.py                          # pulls games, teams, talent, and SP+ data from the CFBD API
├── rank_teams.py                   # runs the Elo model over the pulled data
├── data/
│   ├── games_<year>.csv            # one file per season
│   ├── games_all_years.csv         # combined historical games
│   ├── teams_<year>.csv            # FBS roster for that season
│   ├── talent_<year>.csv           # recruiting talent composite by team
│   ├── sp_ratings_<year>.csv       # SP+ ratings (external comparison)
│   ├── elo_ratings_weekly.csv      # every team's rating after every week it played
│   └── elo_ratings_season_final.csv # final (or current, for the in-progress season) rating per team per season
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

**1. Get a free API key** at [collegefootballdata.com](https://collegefootballdata.com).

**2. Install the one dependency:**

```bash
pip install requests
```

**3. Set your API key as an environment variable** (never hardcode it in the
scripts or commit it to source control):

```bash
# macOS/Linux
export CFBD_API_KEY="your_key_here"

# Windows PowerShell
$env:CFBD_API_KEY="your_key_here"
```

## Usage

**Pull data** (defaults to seasons 2019–2024; pass your own range):

```bash
python cfb.py --start-year 2019 --end-year 2026
```

**Run the ranking engine** on whatever's in `data/`:

```bash
python rank_teams.py
```

This prints the current Top 25 to the console and writes the full weekly and
season-final ratings to `data/elo_ratings_weekly.csv` and
`data/elo_ratings_season_final.csv`.

**Keeping current-season rankings up to date:** re-run `cfb.py` for the
current year (e.g. `python cfb.py --start-year 2026 --end-year 2026`) each
week as new games are completed, then re-run `rank_teams.py`.

## Roadmap

- [ ] Backtest ranking accuracy against historical AP Poll and SP+ ratings
- [ ] Automate weekly data pulls and rating updates via a scheduled GitHub Action
- [ ] Build a simple dashboard to visualize current rankings and a team's
      rating trajectory across a season
- [ ] Publish a "why is this team ranked here" breakdown showing the
      schedule-strength math behind each rating

## License

MIT — see [LICENSE](LICENSE). Contributions, forks, and questions are welcome;
this project is meant to be learned from as much as used.