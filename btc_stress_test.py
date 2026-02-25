"""
btc_stress_test.py
==================
Brutal audit of the Sentinel Adaptive Exposure system.

Friction Rules:
- 0.10% (10 bps) Commission per trade notional.
- 0.02% (2 bps) Slippage penalty per trade.
- ZERO LOOKAHEAD: Exposure for bar[i] is decided by labels[i-1].
- NO COMPOUNDING ADVANTAGE: All checks use standard sequential logic.
"""

import datetime
import statistics
import collections
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 0. CONFIG & DATA
# ============================================================================
NPY_PATH = r"C:\Users\yashw\Projects\Sentinel\experiments\btc_close_clean.npy"
BAR_MIN  = 15
START_DT = datetime.datetime(2018, 1, 1)

# Detection Parameters (Frozen)
SMA_WIN       = 19_200 
ENTER_PCT     = 0.10   
EXIT_PCT      = 0.04   
DRIFT_CONFIRM = 3      

FEES     = 0.0010  # 0.1% per trade notional
SLIPPAGE = 0.0002  # 0.02% per trade notional

def get_signal_and_labels(prices):
    N = len(prices)
    sma = np.full(N, np.nan)
    win_sum = 0.0
    for i in range(N):
        win_sum += prices[i]
        if i >= SMA_WIN: win_sum -= prices[i - SMA_WIN]
        if i >= SMA_WIN - 1: sma[i] = win_sum / SMA_WIN
    
    signal = np.where(np.isnan(sma), np.nan, (prices - sma) / sma)
    state = "NORMAL"
    e_up = e_dn = ex_cnt = 0
    labels = []

    for sig in signal:
        if np.isnan(sig):
            labels.append("NORMAL")
            continue
        if state == "NORMAL":
            if sig > ENTER_PCT: e_up += 1; e_dn = 0
            elif sig < -ENTER_PCT: e_dn += 1; e_up = 0
            else: e_up = e_dn = 0
            if e_up >= DRIFT_CONFIRM:
                state = "UP"; label = "UP"
                for j in range(1, DRIFT_CONFIRM):
                    if len(labels) >= j and labels[-j] == "NORMAL": labels[-j] = "UP"
            elif e_dn >= DRIFT_CONFIRM:
                state = "DOWN"; label = "DOWN"
                for j in range(1, DRIFT_CONFIRM):
                    if len(labels) >= j and labels[-j] == "NORMAL": labels[-j] = "DOWN"
            else: label = "NORMAL"
        elif state == "UP":
            if sig < EXIT_PCT:
                ex_cnt += 1
                if ex_cnt >= DRIFT_CONFIRM:
                    state = "NORMAL"; ex_cnt = e_up = e_dn = 0; label = "NORMAL"
                else: label = "UP"
            else: ex_cnt = 0; label = "UP"
        else: # DOWN
            if sig > -EXIT_PCT:
                ex_cnt += 1
                if ex_cnt >= DRIFT_CONFIRM:
                    state = "NORMAL"; ex_cnt = e_up = e_dn = 0; label = "NORMAL"
                else: label = "DOWN"
            else: ex_cnt = 0; label = "DOWN"
        labels.append(label)
    return signal, labels

def calc_metrics(equity):
    ret = (equity[-1] / equity[0] - 1) * 100
    years = len(equity) / (365.25 * 96)
    cagr = ((equity[-1] / equity[0]) ** (1/years) - 1) * 100
    peak = 1e-9
    mdd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > mdd: mdd = dd
    return ret, cagr, mdd * 100

# ============================================================================
# 1. BRUTAL SIMULATION (CAUSAL + FRICTION)
# ============================================================================
prices = np.load(NPY_PATH)
N = len(prices)
timestamps = [START_DT + datetime.timedelta(minutes=BAR_MIN * i) for i in range(N)]
idx_2019 = next(i for i, t in enumerate(timestamps) if t >= datetime.datetime(2019, 1, 1))
idx_2023 = next(i for i, t in enumerate(timestamps) if t >= datetime.datetime(2023, 1, 1))

signal, labels = get_signal_and_labels(prices)

def simulate_with_friction(p_subset, l_subset, s_subset):
    equity = [1.0]
    last_major = "DOWN"
    prev_exposure = 0.0
    
    # We start from bar 1. 
    # Exposure for bar 'i' is derived from label/signal at 'i-1'
    for i in range(1, len(l_subset)):
        # 1. Causal Logic: Look back at previous bar
        l_prev = l_subset[i-1]
        s_prev = s_subset[i-1]
        
        # Determine Exposure based on signal at CLOSE of previous bar
        if l_prev == "UP":
            exposure = min(1.0, abs(s_prev) / (ENTER_PCT * 1.5))
            last_major = "UP"
        elif l_prev == "DOWN":
            exposure = 0.0
            last_major = "DOWN"
        else: # NORMAL
            exposure = 0.6 if last_major == "UP" else 0.2
            
        # 2. Friction: Cost of adjusting exposure
        # Applied to current equity BEFORE the price move
        trade_size = abs(exposure - prev_exposure)
        cost = equity[-1] * trade_size * (FEES + SLIPPAGE)
        equity_after_friction = equity[-1] - cost
        
        # 3. Price Move
        # Benefit from price move of bar 'i' (p[i-1] to p[i])
        price_ret = p_subset[i] / p_subset[i-1]
        equity_next = equity_after_friction * (1.0 + exposure * (price_ret - 1.0))
        
        equity.append(equity_next)
        prev_exposure = exposure
        
    return np.array(equity)

# Execution
f_p, f_l, f_s = prices[idx_2019:], labels[idx_2019:], signal[idx_2019:]
eq_brutal = simulate_with_friction(f_p, f_l, f_s)
eq_bh     = f_p / f_p[0]

# Splits
split_off = idx_2023 - idx_2019

def audit(name, eq_full, split_i):
    full_met = calc_metrics(eq_full)
    fwd_met  = calc_metrics(eq_full[split_i:] / eq_full[split_i])
    print(f"\n--- {name} ---")
    print(f"FULL:    Ret {full_met[0]:7.1f}% | CAGR {full_met[1]:5.1f}% | MDD {full_met[2]:4.1f}%")
    print(f"FORWARD: Ret {fwd_met[0]:7.1f}% | CAGR {fwd_met[1]:5.1f}% | MDD {fwd_met[2]:4.1f}%")

audit("BUY & HOLD (Benchmark)", eq_bh, split_off)
audit("ADAPTIVE SENTINEL (STRESS TEST)", eq_brutal, split_off)
