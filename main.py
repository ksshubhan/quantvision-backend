from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import datetime

app = FastAPI()

# --- CORS setup ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://quantvision.vercel.app",
    "https://quantvision-sshubhans-projects.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Test route ---
@app.get("/")
def read_root():
    return {"message": "Backend is running successfully with CORS!"}

# --- Simple backtest route ---
@app.get("/run_strategy")
def run_strategy(name: str = "momentum", ticker: str = "AAPL"):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import math

    # 1. Download real stock data
    df = yf.download(ticker, period="6mo", interval="1d")
    if df.empty:
        return {"error": "Invalid ticker or no data found."}

    # 2. Calculate returns
    df["returns"] = df["Close"].pct_change()

    # 3. Generate strategy signals
    if name == "momentum":
        df["signal"] = np.where(df["returns"] > 0, 1, -1)
    elif name == "mean_reversion":
        df["signal"] = np.where(df["returns"] < 0, 1, -1)
    elif name == "sma_crossover":
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["signal"] = np.where(df["SMA20"] > df["SMA50"], 1, -1)
    else:
        df["signal"] = 1

    # 4. Calculate equity curve
    df["strategy_return"] = df["signal"].shift(1) * df["returns"]
    df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod() * 100

    # 5. Compute metrics safely
    sharpe = np.sqrt(252) * (df["strategy_return"].mean() / df["strategy_return"].std()) if df["strategy_return"].std() != 0 else 0
    drawdown = float((1 - df["equity_curve"] / df["equity_curve"].cummax()).max() * 100)
    annual_ret = float(((1 + df["strategy_return"].mean()) ** 252 - 1) * 100)
    vol = float(df["strategy_return"].std() * np.sqrt(252) * 100)

    metrics = {
        "sharpe_ratio": round(float(sharpe), 2) if not math.isnan(sharpe) else 0,
        "max_drawdown": round(drawdown, 2),
        "annual_return": round(annual_ret, 2),
        "volatility": round(vol, 2),
    }

    # --- JSON-safe output using tuples ---
    df = df.reset_index()
    df["date"] = df["Date"].astype(str)
    df["portfolio_value"] = df["equity_curve"].astype(float)

    # Create list of tuples: [(date, value), (date, value), ...]
    chart_data = list(zip(df["date"].tolist(), df["portfolio_value"].tolist()))

    # Return as JSON
    return {"equity_curve": chart_data, "metrics": metrics}


