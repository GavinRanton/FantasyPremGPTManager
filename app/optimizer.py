from ortools.sat.python import cp_model
import pandas as pd
from typing import Dict, Tuple

MAX_FROM_CLUB = 3

def pick_initial_squad(players: pd.DataFrame, budget_m: float, bench_weight: float=0.10) -> Tuple[pd.DataFrame, int, int]:
    # players must include: id, pos, club, price_m, EP_next
    model = cp_model.CpModel()
    idxs = list(players.index)
    # decision vars
    x = {i: model.NewBoolVar(f"x_{i}") for i in idxs}   # selected in 15
    st = {i: model.NewBoolVar(f"st_{i}") for i in idxs} # starter in XI
    cpt = {i: model.NewBoolVar(f"cpt_{i}") for i in idxs}
    vice = {i: model.NewBoolVar(f"vice_{i}") for i in idxs}

    # Budget
    model.Add(sum(int(round(players.loc[i, 'price_m']*10))*x[i] for i in idxs) <= int(round(budget_m*10)))

    # Position counts in 15
    def count_pos(pos): 
        return sum(x[i] for i in idxs if players.loc[i,"pos"] == pos)

    model.Add(count_pos("GKP") == 2)
    model.Add(count_pos("DEF") == 5)
    model.Add(count_pos("MID") == 5)
    model.Add(count_pos("FWD") == 3)

    # Club limit
    clubs = players["club"].unique().tolist()
    for club in clubs:
        model.Add(sum(x[i] for i in idxs if players.loc[i,"club"] == club) <= MAX_FROM_CLUB)

    # Starters: 11, exactly 1 GK starter, min DEF/MID/FWD constraints
    model.Add(sum(st[i] for i in idxs) == 11)
    model.Add(sum(st[i] for i in idxs if players.loc[i,"pos"] == "GKP") == 1)
    model.Add(sum(st[i] for i in idxs if players.loc[i,"pos"] == "DEF") >= 3)
    model.Add(sum(st[i] for i in idxs if players.loc[i,"pos"] == "MID") >= 2)
    model.Add(sum(st[i] for i in idxs if players.loc[i,"pos"] == "FWD") >= 1)

    # Logical: starter implies selected
    for i in idxs:
        model.Add(st[i] <= x[i])

    # Captain/Vice: both among starters, unique
    model.Add(sum(cpt[i] for i in idxs) == 1)
    model.Add(sum(vice[i] for i in idxs) == 1)
    for i in idxs:
        model.Add(cpt[i] <= st[i])
        model.Add(vice[i] <= st[i])
        model.Add(cpt[i] + vice[i] <= 1)

    # Objective: starters EP + captain double + small bench weight
    ep = {i: float(players.loc[i, "EP_next"]) for i in idxs}
    obj = []
    obj += [ep[i]*st[i] for i in idxs]
    obj += [ep[i]*cpt[i]]  # extra EP for captain (double counts)
    obj += [bench_weight*ep[i]*(x[i] - st[i]) for i in idxs]
    model.Maximize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Optimizer failed to find a solution")

    picked = players[[
        "id","display_name","pos","club","price_m","EP_next"
    ]].copy()
    picked["selected"] = [int(solver.Value(x[i])) for i in idxs]
    picked["starter"] = [int(solver.Value(st[i])) for i in idxs]
    picked["is_captain"] = [int(solver.Value(cpt[i])) for i in idxs]
    picked["is_vice"] = [int(solver.Value(vice[i])) for i in idxs]

    cap_id = int(picked.loc[picked["is_captain"]==1, "id"].head(1).values[0])
    vice_id = int(picked.loc[picked["is_vice"]==1, "id"].head(1).values[0])
    team = picked[picked["selected"]==1].copy()

    # Bench order: GK2 first, then by ascending EP among non-starters
    bench = team[team["starter"]==0].copy()
    gk = bench[bench["pos"]=="GKP"].copy()
    of = bench[bench["pos"]!="GKP"].copy().sort_values("EP_next")
    bench_order = pd.concat([gk, of], ignore_index=True)
    team = team.merge(bench_order.reset_index().rename(columns={"index":"bench_order"}), how="left")

    team.rename(columns={"display_name":"name"}, inplace=True)
    return team, cap_id, vice_id

def best_lineup(team_df: pd.DataFrame) -> pd.DataFrame:
    # pick XI: 1 GK + best 10 outfielders with formation constraints (min DEF 3, MID 2, FWD 1)
    gk = team_df[team_df["pos"]=="GKP"].sort_values("EP_next", ascending=False).head(1)
    out = team_df[team_df["pos"]!="GKP"].copy()
    # Greedy: start by top EP, then fix formation minima
    out_sorted = out.sort_values("EP_next", ascending=False)
    start = []
    counts = {"DEF":0,"MID":0,"FWD":0}
    for _, r in out_sorted.iterrows():
        pos = r["pos"]; start.append(r)
        counts[pos] += 1
        if len(start) == 10: break
    # enforce minima by swapping if needed
    def ensure_min(pos, k):
        nonlocal start, counts
        if counts[pos] >= k: return
        # find from bench with highest EP of pos
        pool = [r for _,r in out_sorted.iterrows() if r["pos"]==pos and r["id"] not in [s["id"] for s in start]]
        if not pool: return
        candidate = pool[0]
        # replace the currently selected with smallest EP from a pos with surplus
        surplus_positions = [p for p,c in counts.items() if (p!=pos and c> (3 if p=="DEF" else 2 if p=="MID" else 1))]
        if not surplus_positions:
            # replace lowest EP overall
            lowest = min(start, key=lambda r: r["EP_next"])
            counts[lowest["pos"]] -= 1
            start.remove(lowest)
        else:
            # replace lowest EP among surplus positions
            lowest = min([r for r in start if r["pos"] in surplus_positions], key=lambda r: r["EP_next"])
            counts[lowest["pos"]] -= 1
            start.remove(lowest)
        start.append(candidate)
        counts[pos] += 1
        ensure_min(pos, k)
    ensure_min("DEF",3); ensure_min("MID",2); ensure_min("FWD",1)
    lineup = pd.concat([gk, pd.DataFrame(start)], ignore_index=True).sort_values("pos", ascending=False)
    return lineup
