from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd

from .config import settings
from . import data as D
from . import ep as EP
from .optimizer import pick_initial_squad, best_lineup
from .planner import plan_week
from .schemas import (
    InitialSquadRequest, WeeklyPlanRequest, WhatIfRequest,
    PlayerPick, SquadResponse, WeeklyPlanResponse, DeadlinesResponse
)

app = FastAPI(title="FPL Manager API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/deadlines", response_model=DeadlinesResponse)
def deadlines():
    boot = D.get_bootstrap()
    ev = D.get_events_from_bootstrap(boot).sort_values("gw")
    take = ev.head(3)
    return DeadlinesResponse(next_events=take.to_dict(orient="records"))

def _build_players_ep(horizon_gws: int):
    pm = D.build_player_master(settings.season)
    fixtures = pd.DataFrame(D.get_fixtures())
    # Keep only PL fixtures (FPL uses code 1)
    if "competition" in fixtures.columns:
        fixtures = fixtures[fixtures["competition"]==1]
    players_ep = EP.sum_horizon_ep(pm, fixtures, horizon_gws)
    # rename for optimizer
    df = players_ep.rename(columns={"team_name":"team_long"})
    return df

@app.post("/initial-squad", response_model=SquadResponse)
def initial_squad(req: InitialSquadRequest):
    players = _build_players_ep(req.horizon_gws)

    # Select candidates: filter unlikely minutes
    players = players[players["price_m"]>0].copy()
    players["EP_next"] = players["EP_next"].fillna(0.0)

    team, cap, vice = pick_initial_squad(players[["id","display_name","pos","club","price_m","EP_next"]], req.budget_m, settings.bench_weight)

    out = []
    for _, r in team.iterrows():
        out.append(PlayerPick(
            id=int(r["id"]),
            name=str(r["display_name"]),
            pos=str(r["pos"]),
            club=str(r["club"]),
            price_m=float(r["price_m"]),
            ep_next=float(r["EP_next"]),
            starter=bool(r["starter"]==1),
            bench_order=int(r["bench_order"]) if pd.notna(r["bench_order"]) else None
        ))
    # compute bank
    spend = team["price_m"].sum()
    bank = max(0.0, req.budget_m - float(spend))
    return SquadResponse(team=out, captain_id=int(cap), vice_id=int(vice), bank_m=float(bank))

@app.post("/weekly-plan", response_model=WeeklyPlanResponse)
def weekly_plan(req: WeeklyPlanRequest):
    players = _build_players_ep(req.horizon_gws)
    transfers, lineup_df, bench_df, ep_gain, hit_cost = plan_week(players[["id","display_name","pos","club","price_m","EP_next"]], req.current_squad, req.bank_m, req.free_transfers)

    def to_pick(r):
        return PlayerPick(
            id=int(r["id"]), name=str(r["display_name"]), pos=str(r["pos"]),
            club=str(r["club"]), price_m=float(r["price_m"]), ep_next=float(r["EP_next"]),
            starter=True, bench_order=None
        )
    lineup = [to_pick(r) for _, r in lineup_df.iterrows()]
    bench = [
        PlayerPick(
            id=int(r["id"]), name=str(r["display_name"]), pos=str(r["pos"]),
            club=str(r["club"]), price_m=float(r["price_m"]), ep_next=float(r["EP_next"]),
            starter=False, bench_order=i+1
        ) for i, (_, r) in enumerate(bench_df.iterrows())
    ]

    # captain & vice: top 2 EP in lineup
    sorted_xi = lineup_df.sort_values("EP_next", ascending=False)
    cpt = int(sorted_xi.iloc[0]["id"])
    vice = int(sorted_xi.iloc[1]["id"]) if len(sorted_xi)>1 else cpt

    notes = []
    if hit_cost>0:
        notes.append(f"Includes hit cost of -{hit_cost} points")
    if ep_gain <= 0:
        notes.append("No positive EP move found; hold transfer recommended")

    return WeeklyPlanResponse(
        transfers=transfers, lineup=lineup, captain_id=cpt, vice_id=vice,
        bench=bench, ep_gain_vs_stand_pat=float(ep_gain), hit_cost=int(hit_cost),
        notes=notes
    )

@app.post("/what-if", response_model=SquadResponse)
def what_if(req: WhatIfRequest):
    players = _build_players_ep(req.horizon_gws)
    team, cap, vice = pick_initial_squad(players[["id","display_name","pos","club","price_m","EP_next"]], req.budget_m, bench_weight=0.05)
    out = []
    for _, r in team.iterrows():
        out.append(PlayerPick(
            id=int(r["id"]), name=str(r["display_name"]), pos=str(r["pos"]),
            club=str(r["club"]), price_m=float(r["price_m"]), ep_next=float(r["EP_next"]),
            starter=bool(r["starter"]==1), bench_order=int(r["bench_order"]) if pd.notna(r["bench_order"]) else None
        ))
    spend = team["price_m"].sum()
    bank = max(0.0, req.budget_m - float(spend))
    return SquadResponse(team=out, captain_id=int(cap), vice_id=int(vice), bank_m=float(bank))
