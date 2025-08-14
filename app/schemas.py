from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ----- Requests -----
class InitialSquadRequest(BaseModel):
    budget_m: float = Field(100.0, ge=90.0, le=120.0)
    horizon_gws: int = Field(3, ge=1, le=6)

class WeeklyPlanRequest(BaseModel):
    current_squad: List[int]  # list of FPL element IDs (15 players)
    bank_m: float = Field(0.0, ge=0.0)
    free_transfers: int = Field(1, ge=0, le=5)
    horizon_gws: int = Field(2, ge=1, le=6)

class WhatIfRequest(BaseModel):
    budget_m: float = Field(100.0, ge=90.0, le=120.0)
    horizon_gws: int = Field(2, ge=1, le=6)
    min_def: int = 3
    min_mid: int = 2
    min_fwd: int = 1

# ----- Responses -----
class PlayerPick(BaseModel):
    id: int
    name: str
    pos: str
    club: str
    price_m: float
    ep_next: float
    starter: bool
    bench_order: Optional[int] = None

class SquadResponse(BaseModel):
    team: List[PlayerPick]
    captain_id: int
    vice_id: int
    bank_m: float

class WeeklyPlanResponse(BaseModel):
    transfers: List[Dict]
    lineup: List[PlayerPick]
    captain_id: int
    vice_id: int
    bench: List[PlayerPick]
    ep_gain_vs_stand_pat: float
    hit_cost: int
    notes: List[str]

class DeadlinesResponse(BaseModel):
    next_events: List[Dict]  # [{gw, deadline_utc, is_next, is_current}]

