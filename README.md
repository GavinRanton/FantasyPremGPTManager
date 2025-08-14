# FPL Manager API (FastAPI + EP model + ILP)

A tiny API your Custom GPT can call to act as your Fantasy Premier League manager.
It pulls live data from the **FPL API** and augments it with the **FPL‑Elo‑Insights** GitHub dataset.
It computes expected points (EP), optimizes your initial squad, and proposes weekly transfers.

## Quick Start

```bash
# 1) Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for Swagger UI.

### Minimal Configuration

Default season: `2025-2026`. You can change the season or base URL in `app/config.py`.

### Endpoints

- `POST /initial-squad` → Optimize a 15-man squad, starting XI, captain/vice, bench order
- `POST /weekly-plan` → Propose transfers for next GW(s) + lineup + captain/vice
- `GET /deadlines` → Next 3 GW deadlines (Europe/London assumed by your GPT)
- `POST /what-if` → Simulate alternative formations/horizons

### GPT Action

Use the provided `openapi.yaml` in **Custom GPT → Actions** and the prompt in `action_instructions.txt`.

### Notes

- This ships a pragmatic EP model that uses xG/xA per 90 when available and falls back to proxies if not.
- Clean sheet probability is derived from team strengths (FPL) and optionally Elo (if available) via a calibrated logistic.
- The optimizer is an ILP using OR‑Tools CP‑SAT and enforces budget, club limits, and valid formations.
- Weekly planning evaluates single and two-transfer moves against free transfers (‑4 per extra transfer).

