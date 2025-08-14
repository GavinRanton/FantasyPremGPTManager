import requests, pandas as pd, numpy as np
from .config import settings

def _get_json(url: str, timeout=settings.request_timeout_sec):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "FPL-Manager-API/1.0"})
    r.raise_for_status()
    return r.json()

def _get_csv(url: str, timeout=settings.request_timeout_sec):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "FPL-Manager-API/1.0"})
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

# -------- FPL API --------
def get_bootstrap():
    return _get_json(f"{settings.fpl_api_base}/bootstrap-static/")

def get_fixtures():
    return _get_json(f"{settings.fpl_api_base}/fixtures/")

def get_events_from_bootstrap(boot):
    # returns DataFrame of events (gameweeks)
    ev = pd.DataFrame(boot.get("events", []))
    keep = ["id", "deadline_time", "is_current", "is_next", "is_previous"]
    return ev[keep].rename(columns={"id":"gw","deadline_time":"deadline_utc"})

# -------- Elo Insights (CSV on GitHub) --------
def get_elo_players(season: str=None):
    season = season or settings.season
    url = f"{settings.elo_repo_base}/{season}/players.csv"
    try:
        return _get_csv(url)
    except Exception:
        return pd.DataFrame()

def get_elo_playerstats(season: str=None):
    season = season or settings.season
    url = f"{settings.elo_repo_base}/{season}/playerstats.csv"
    try:
        return _get_csv(url)
    except Exception:
        return pd.DataFrame()

def get_elo_teams(season: str=None):
    season = season or settings.season
    url = f"{settings.elo_repo_base}/{season}/teams.csv"
    try:
        return _get_csv(url)
    except Exception:
        return pd.DataFrame()

# -------- Unified Player Table --------
POS_MAP = {1:"GKP", 2:"DEF", 3:"MID", 4:"FWD"}

def build_player_master(season: str=None):
    season = season or settings.season
    boot = get_bootstrap()
    elements = pd.DataFrame(boot.get("elements", []))
    teams = pd.DataFrame(boot.get("teams", []))[["id","name","short_name","strength_overall_home","strength_overall_away","strength_attack_home","strength_attack_away","strength_defence_home","strength_defence_away"]]

    df = elements[["id","web_name","first_name","second_name","team","element_type","now_cost","status","chance_of_playing_next_round","chance_of_playing_this_round","news","minutes","ict_index","selected_by_percent"]].copy()
    df["pos"] = df["element_type"].map(POS_MAP)
    df["price_m"] = df["now_cost"].fillna(0).astype(float) / 10.0

    df = df.merge(teams, left_on="team", right_on="id", how="left", suffixes=("","_team"))
    df.rename(columns={"name":"team_name","short_name":"club"}, inplace=True)

    # Join elo playerstats if available
    pstats = get_elo_playerstats(season)
    if not pstats.empty:
        # Try to join on player name (best-effort). Expect columns like 'player_name','xG90','xA90','CBIT90' etc.
        # Normalize names
        def norm(s):
            return str(s).strip().lower()
        df["name_key"] = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.strip()
        df["name_key"] = df["name_key"].where(df["name_key"].str.len()>1, df["web_name"])
        df["name_key"] = df["name_key"].apply(norm)

        if "player_name" in pstats.columns:
            pstats["name_key"] = pstats["player_name"].astype(str).str.lower().str.strip()
        elif "name" in pstats.columns:
            pstats["name_key"] = pstats["name"].astype(str).str.lower().str.strip()
        else:
            pstats["name_key"] = pstats.iloc[:,0].astype(str).str.lower().str.strip()

        # Standardize likely stat columns
        for col in ["xG90","xA90","xGI90","starts_last6","mins_last6"]:
            if col not in pstats.columns:
                pstats[col] = np.nan

        df = df.merge(pstats[["name_key","xG90","xA90","xGI90","starts_last6","mins_last6"]], on="name_key", how="left")

    df["display_name"] = df["web_name"]
    return df

