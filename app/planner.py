import pandas as pd, itertools
from typing import Dict, List, Tuple
from .optimizer import best_lineup

def plan_week(players_all: pd.DataFrame, current_ids: List[int], bank_m: float, free_transfers: int, max_swaps_considered:int=2):
    team = players_all[players_all["id"].isin(current_ids)].copy()

    # compute baseline lineup EP
    lineup0 = best_lineup(team)
    ep0 = lineup0["EP_next"].sum()

    # Candidate pool: top 40 EP players not in team
    pool = players_all[~players_all["id"].isin(current_ids)].sort_values("EP_next", ascending=False).head(40).copy()

    # Helper: check constraints (max 3 per club, budget)
    def valid_after(out_ids: List[int], in_ids: List[int]) -> bool:
        t = team.copy()
        t = t[~t["id"].isin(out_ids)]
        t = pd.concat([t, players_all[players_all["id"].isin(in_ids)]], ignore_index=True)
        # club constraint
        if (t.groupby("club")["id"].count() > 3).any():
            return False
        # position counts in 15
        if (t[t["pos"]=="GKP"].shape[0] != 2 or t[t["pos"]=="DEF"].shape[0] > 5 or
            t[t["pos"]=="MID"].shape[0] > 5 or t[t["pos"]=="FWD"].shape[0] > 3):
            # Allow equal or fewer before we fill; but initial squads should already match.
            pass
        # budget
        price_out = team[team["id"].isin(out_ids)]["price_m"].sum()
        price_in  = players_all[players_all["id"].isin(in_ids)]["price_m"].sum()
        delta = price_in - price_out
        return delta <= bank_m + 1e-6

    best = {"out":[],"in":[],"ep_gain":0.0,"hit":0}
    # Single transfers
    for _, out in team.iterrows():
        for _, inr in pool.iterrows():
            if out["pos"] != inr["pos"]:
                continue
            if not valid_after([out["id"]],[inr["id"]]):
                continue
            t1 = team[team["id"]!=out["id"]].copy()
            t1 = pd.concat([t1, inr.to_frame().T], ignore_index=True)
            ep1 = best_lineup(t1)["EP_next"].sum()
            hit = 0 if free_transfers >= 1 else 4
            gain = ep1 - ep0 - hit
            if gain > best["ep_gain"]:
                best = {"out":[int(out["id"])], "in":[int(inr["id"])], "ep_gain": float(gain), "hit": hit}

    # Two transfers (restrict search to top 8 pool per position for speed)
    pool_small = pool.groupby("pos").head(8)
    outs = team.copy()
    for (i1, out1) in outs.iterrows():
        for (i2, out2) in outs.iterrows():
            if out2["id"] <= out1["id"]: 
                continue
            # choose ins of the same positions for simplicity
            cand_in1 = pool_small[pool_small["pos"]==out1["pos"]]
            cand_in2 = pool_small[pool_small["pos"]==out2["pos"]]
            for _, in1 in cand_in1.iterrows():
                for _, in2 in cand_in2.iterrows():
                    in_ids = [int(in1["id"]), int(in2["id"])]
                    out_ids = [int(out1["id"]), int(out2["id"])]
                    if not valid_after(out_ids, in_ids):
                        continue
                    t2 = team[~team["id"].isin(out_ids)].copy()
                    t2 = pd.concat([t2, in1.to_frame().T, in2.to_frame().T], ignore_index=True)
                    ep2 = best_lineup(t2)["EP_next"].sum()
                    needed = max(0, 2 - free_transfers)
                    hit = 4*needed
                    gain = ep2 - ep0 - hit
                    if gain > best["ep_gain"]:
                        best = {"out": out_ids, "in": in_ids, "ep_gain": float(gain), "hit": hit}

    # Build recommendation
    if best["ep_gain"] <= 0:
        rec_transfers = []
        tbest = team
        lineup = best_lineup(tbest)
        bench = tbest[~tbest["id"].isin(lineup["id"])].copy().sort_values("EP_next")
        return rec_transfers, lineup, bench, 0.0, 0

    # Apply best transfers to return proposed team
    tnew = team[~team["id"].isin(best["out"])].copy()
    tnew = pd.concat([tnew, players_all[players_all['id'].isin(best['in'])]], ignore_index=True)
    lineup = best_lineup(tnew)
    bench = tnew[~tnew["id"].isin(lineup["id"])].copy().sort_values("EP_next")

    transfers = []
    for out_id in best["out"]:
        out_row = team[team["id"]==out_id].iloc[0]
        transfers.append({"out_id": int(out_id), "out_name": str(out_row["display_name"]), "out_pos": str(out_row["pos"]), "out_price_m": float(out_row["price_m"])})
    for i, in_id in enumerate(best["in"]):
        in_row = players_all[players_all["id"]==in_id].iloc[0]
        transfers[i]["in_id"] = int(in_id)
        transfers[i]["in_name"] = str(in_row["display_name"])
        transfers[i]["in_pos"] = str(in_row["pos"])
        transfers[i]["in_price_m"] = float(in_row["price_m"])

    return transfers, lineup, bench, float(best["ep_gain"]), int(best["hit"])
