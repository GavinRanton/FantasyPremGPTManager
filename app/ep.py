import numpy as np, pandas as pd, math
from .config import settings

GOAL_POINTS = {"GKP":6, "DEF":6, "MID":5, "FWD":4}
ASSIST_POINTS = 3
CS_POINTS = {"GKP":4, "DEF":4, "MID":1, "FWD":0}

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def minutes_prob(row):
    # Use FPL 'chance_of_playing_next_round' if present; else infer from status
    cop = row.get("chance_of_playing_next_round", None)
    status = row.get("status","a")
    if pd.notna(cop):
        return float(cop)/100.0
    return {"a":0.92, "d":0.6, "i":0.1, "n":0.0, "s":0.0}.get(status, 0.8)

def expected_minutes(row):
    # Scale by starts/mins last 6 if available
    mp = minutes_prob(row)
    mins_recent = row.get("mins_last6", np.nan)
    if pd.notna(mins_recent):
        # average minutes in last 6 appearances (rough)
        avg = float(mins_recent)/max(1.0, row.get("starts_last6", 3))
        avg = max(30.0, min(90.0, avg))
    else:
        avg = 75.0
    return mp * avg

def clean_sheet_prob(row, opp_strength_att: float, our_strength_def: float, venue: str):
    # Use team strengths; adjust by venue. This is a calibrated logistic proxy.
    # Higher our_def and lower opp_att -> higher CS probability.
    ven_adj = 1.0 if venue == "H" else -1.0
    x = (our_strength_def - opp_strength_att) * 0.015 + (0.25 * ven_adj)
    return max(0.02, min(0.75, sigmoid(x)))

def attack_rates(row):
    # Pull xG/xA per 90 if available; otherwise proxy from ICT index
    xg90 = row.get("xG90", np.nan)
    xa90 = row.get("xA90", np.nan)
    if pd.isna(xg90) and pd.isna(xa90):
        ict = float(row.get("ict_index", 0.0))
        # crude proxy: distribute ICT to xG+xA
        xg90 = 0.02 * ict
        xa90 = 0.015 * ict
    return max(0.0, float(xg90 or 0.0)), max(0.0, float(xa90 or 0.0))

def ep_for_fixture(row, venue: str, opp_att: float, our_def: float):
    pos = row["pos"]
    mins = expected_minutes(row)
    # Appearance points: approximate 2 if plays >=60; scale by probability
    p60 = min(1.0, mins/60.0)
    appearance = 2.0 * p60

    # Clean sheet
    pcs = clean_sheet_prob(row, opp_att, our_def, venue)
    cs_pts = CS_POINTS.get(pos, 0) * pcs

    # Attack
    xg90, xa90 = attack_rates(row)
    g_pts = GOAL_POINTS.get(pos, 4)
    attack = (mins/90.0) * (xg90 * g_pts + xa90 * ASSIST_POINTS)

    # Light bonus prior
    bonus = 0.25 if pos in ("MID","FWD") else 0.15

    return appearance + cs_pts + attack + bonus

def sum_horizon_ep(players_df: pd.DataFrame, fixtures_df: pd.DataFrame, horizon_gws: int):
    # Build quick lookups for team strengths
    team_strength = {}
    # strengths are per team, both home/away def/att
    for _, r in players_df.drop_duplicates("team").iterrows():
        team_strength[int(r["team"])] = {
            "att_home": float(r.get("strength_attack_home", 1000.0)),
            "att_away": float(r.get("strength_attack_away", 1000.0)),
            "def_home": float(r.get("strength_defence_home", 1000.0)),
            "def_away": float(r.get("strength_defence_away", 1000.0)),
        }

    # Determine next N events
    fut = fixtures_df[fixtures_df["event"].notna()].copy()
    if fut.empty:
        # fallback: use all fixtures
        fut = fixtures_df.copy()
    next_events = sorted([int(e) for e in fut["event"].dropna().unique()])[:horizon_gws]

    # Precompute per-team sequence of opponents (first N upcoming fixtures)
    team_fixtures = {int(t): [] for t in players_df["team"].unique()}
    for _, fx in fut[fut["event"].isin(next_events)].iterrows():
        h = int(fx["team_h"]); a = int(fx["team_a"])
        team_fixtures[h].append(("H", int(fx["team_a"])))
        team_fixtures[a].append(("A", int(fx["team_h"])))

    ep_list = []
    for idx, row in players_df.iterrows():
        t = int(row["team"])
        fixtures = team_fixtures.get(t, [])[:horizon_gws]
        total = 0.0
        for venue, opp in fixtures:
            ts = team_strength.get(t, None)
            os = team_strength.get(opp, None)
            if not ts or not os:
                continue
            opp_att = os["att_home"] if venue == "A" else os["att_away"]
            our_def = ts["def_home"] if venue == "H" else ts["def_away"]
            total += ep_for_fixture(row, venue, opp_att, our_def)
        ep_list.append(total)

    players_df = players_df.copy()
    players_df["EP_sum"] = ep_list
    # Also compute EP for next GW only (helpful in weekly plans)
    players_df["EP_next"] = players_df["EP_sum"] / max(1, horizon_gws)
    return players_df
