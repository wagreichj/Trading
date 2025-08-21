
#!/usr/bin/env python3
import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.tz import gettz
from ib_insync import IB, util, Stock, MarketOrder

# =========================
# CONFIG (edit as needed)
# =========================
IB_HOST      = "127.0.0.1"
IB_PORT      = 4002            # 4002=paper, 7496=live
IB_CLIENT_ID = DUM518451

TZ      = gettz("America/New_York")
RUN_AT  = time(13, 32)         # 3:45pm ET
UNIVERSE = ["TQQQ","SQQQ","SPY","BSV","UVXY","UPRO"]  # tradeable tickers

# Strategy thresholds (FTLT RSI14 79/31)
RSI_OB       = 79
RSI_OS_TQQQ  = 31
RSI_OS_SPY   = 30

# Risk & execution
TARGET_ALLOCATION = 0.98       # invest 98% of equity in target
MIN_TRADE_USD     = 500.0      # skip tiny trades
USE_RTH           = True       # use regular trading hours for history

# History lookback
LOOKBACK_YEARS    = 6          # enough for 200-SMA stability

# Files
STATE_FILE        = Path("./ftlt_state.json")
LOG_FILE          = Path("./ftlt_log.txt")

# =========================
# Helpers
# =========================
def log(msg: str):
    stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out

async def fetch_daily_closes(ib: IB, symbol: str, years: int) -> pd.Series:
    c = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(c)
    bars = await ib.reqHistoricalDataAsync(
        c,
        endDateTime="",
        durationStr=f"{years} Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=USE_RTH,
        formatDate=1
    )
    if not bars:
        return pd.Series(dtype=float)
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    close = df["close"].astype(float)
    close.index = close.index.tz_localize(None)
    return close

async def build_market_frame(ib: IB) -> pd.DataFrame:
    closes = {}
    for t in UNIVERSE:
        ser = await fetch_daily_closes(ib, t, LOOKBACK_YEARS)
        if ser.empty:
            log(f"⚠️  No history for {t}, skipping.")
            continue
        closes[t] = ser
    if not closes:
        raise RuntimeError("No usable price history downloaded.")
    # Align on intersection of available dates
    idx = None
    for s in closes.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = pd.DatetimeIndex(sorted(idx))
    return pd.DataFrame({t: closes[t].reindex(idx) for t in closes}).dropna(how="any")

def decide_signal(close: pd.DataFrame) -> str | None:
    """
    Implement FTLT RSI14 79/31 decision **based on the last completed bar**.
    Returns one of: 'TQQQ','UVXY','UPRO','SQQQ','BSV' or None (flat).
    """
    must = ["TQQQ","SPY"]
    for m in must:
        if m not in close.columns:  # must have these
            return None

    tqqq = close["TQQQ"]
    spy  = close["SPY"]
    sqqq = close.get("SQQQ")
    bsv  = close.get("BSV")
    uvxy = close.get("UVXY")
    upro = close.get("UPRO")

    tqqq_sma20 = tqqq.rolling(20).mean()
    spy_sma200 = spy.rolling(200).mean()
    rsi_t14    = rsi(tqqq, 14)
    rsi_spy10  = rsi(spy, 10)

    d = close.index[-1]  # yesterday (last completed bar)
    if pd.isna(spy_sma200.loc[d]) or pd.isna(tqqq_sma20.loc[d]):
        return "TQQQ"  # warmup default

    bull = spy.loc[d] > spy_sma200.loc[d]
    pick = None
    if bull:
        if (rsi_t14.loc[d] >= RSI_OB) and (uvxy is not None) and pd.notna(uvxy.loc[d]):
            pick = "UVXY"
        else:
            pick = "TQQQ"
    else:
        if rsi_t14.loc[d] <= RSI_OS_TQQQ:
            pick = "TQQQ"
        elif rsi_spy10.loc[d] <= RSI_OS_SPY and (upro is not None) and pd.notna(upro.loc[d]):
            pick = "UPRO"
        else:
            below20 = tqqq.loc[d] < tqqq_sma20.loc[d]
            if below20:
                # Quick RSI10 compare (last ~30 obs) for SQQQ vs BSV
                def last_rsi10(series: pd.Series | None) -> float:
                    if series is None: return np.nan
                    s = series.dropna().tail(30)
                    return np.nan if s.empty else rsi(s, 10).iloc[-1]
                rs_s = last_rsi10(sqqq)
                rs_b = last_rsi10(bsv)
                if np.isnan(rs_s) and np.isnan(rs_b): pick = "SQQQ" if sqqq is not None else None
                elif np.isnan(rs_b): pick = "SQQQ"
                elif np.isnan(rs_s): pick = "BSV"
                else: pick = "SQQQ" if rs_s >= rs_b else "BSV"
            else:
                pick = "TQQQ"

    return pick

async def account_equity(ib: IB) -> float:
    await ib.reqAccountSummaryAsync()
    vals = [v for v in ib.accountSummary() if v.tag == "NetLiquidation" and v.currency == "USD"]
    return float(vals[0].value) if vals else 0.0

async def last_price(ib: IB, symbol: str) -> float:
    c = Stock(symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(c)
    t = await ib.reqMktDataAsync(c, "", False, False)
    px = t.midpoint() or (t.last or t.close) or 0.0
    return float(px or 0.0)

async def positions_map(ib: IB) -> dict[str, float]:
    pos = await ib.reqPositionsAsync()
    acc = {}
    for p in pos:
        if p.contract.currency != "USD": continue
        sym = p.contract.symbol
        acc[sym] = acc.get(sym, 0.0) + p.position
    return acc

@dataclass
class BotState:
    last_target: str | None = None
    last_trade_day: str | None = None  # "YYYY-MM-DD"

    @classmethod
    def load(cls) -> "BotState":
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                return cls(**d)
            except Exception:
                pass
        return cls()

    def save(self):
        STATE_FILE.write_text(json.dumps(self.__dict__, indent=2))

async def rebalance(ib: IB, target_symbol: str, dry_run=False):
    eq = await account_equity(ib)
    if eq <= 0:
        log("⚠️  Could not read account equity.")
        return

    # Current positions
    pos = await positions_map(ib)

    # Liquidate unwanted symbols (only within our universe)
    for sym, qty in pos.items():
        if sym == target_symbol or sym not in UNIVERSE or qty == 0:
            continue
        side = "SELL" if qty > 0 else "BUY"
        if dry_run:
            log(f"[DRY] Close {sym}: {qty:+.0f} @ MKT")
            continue
        c = Stock(sym, "SMART", "USD")
        await ib.qualifyContractsAsync(c)
        trade = ib.placeOrder(c, MarketOrder(side, abs(int(qty))))
        await trade.fulfilled()
        log(f"Closed {sym}: {qty:+.0f} @ MKT")

    # Compute buy qty for target
    px = await last_price(ib, target_symbol)
    if px <= 0:
        log(f"⚠️  No price for {target_symbol}; skipping buy.")
        return

    # Recompute current target exposure
    pos = await positions_map(ib)
    held_qty = pos.get(target_symbol, 0.0)
    held_val = held_qty * px

    desired_val = eq * TARGET_ALLOCATION
    diff = desired_val - held_val
    if abs(diff) < MIN_TRADE_USD:
        log(f"✓ Rebalance under {MIN_TRADE_USD:.0f} USD; skip.")
        return

    qty = int(abs(diff) // px)  # whole shares
    if qty == 0:
        log("✓ Computed qty=0 after rounding; skip.")
        return

    side = "BUY" if diff > 0 else "SELL"
    if dry_run:
        log(f"[DRY] {side} {qty} {target_symbol} @ MKT (px≈{px:.2f})")
        return

    c = Stock(target_symbol, "SMART", "USD")
    await ib.qualifyContractsAsync(c)
    trade = ib.placeOrder(c, MarketOrder(side, qty))
    await trade.fulfilled()
    log(f"{side} {qty} {target_symbol} @ MKT (px≈{px:.2f})")

def now_et() -> datetime:
    return datetime.now(TZ)

def next_run_time(now: datetime) -> datetime:
    # Next weekday at RUN_AT
    d = now.date()
    run_dt = datetime.combine(d, RUN_AT, tzinfo=TZ)
    if now >= run_dt:
        run_dt = run_dt + timedelta(days=1)
    while run_dt.weekday() >= 5:  # 5=Sat, 6=Sun
        run_dt += timedelta(days=1)
    return run_dt

async def trade_once(ib: IB, dry_run=False):
    # Build market frame & decide target from yesterday's close
    close = await build_market_frame(ib)
    target = decide_signal(close)
    log(f"Signal target: {target}")
    if target is None:
        log("No target (flat); nothing to trade.")
        return

    # State guard to avoid duplicate same-day trades
    st = BotState.load()
    today = now_et().date().isoformat()
    if st.last_trade_day == today and st.last_target == target:
        log("✓ Already traded same target today; skip.")
        return

    await rebalance(ib, target, dry_run=dry_run)

    # Save state
    st
