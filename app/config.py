from dataclasses import dataclass

@dataclass
class Settings:
    # Data sources
    season: str = "2025-2026"
    elo_repo_base: str = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data"
    fpl_api_base: str = "https://fantasy.premierleague.com/api"

    # Modeling
    default_horizon_gws: int = 3  # lookahead for EP sums
    bench_weight: float = 0.10    # small weight for bench EP
    ep_hit_threshold: float = 6.0 # suggest hits only if EP gain >= this (configurable)

    # Safety
    request_timeout_sec: int = 20

settings = Settings()
