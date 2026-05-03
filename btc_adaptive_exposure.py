"""
btc_adaptive_exposure.py
========================

Exposure Logic (Adaptive Sentinel):
• DRIFT_UP: Scale based on strength -> min(1.0, signal / (ENTER_PCT * 1.5))
  Entry floor is ~0.67 exposure at exact threshold crossing (10% / 15%).
• NORMAL:
  - After UP:   60% exposure (Participation in mid-cycle consolidation)
  - After DOWN: 20% exposure (Defensive positioning during bottoming)
• DRIFT_DOWN: 0% exposure (Zero tolerance for bearish drift)

Metrics: FULL (2019-2025) and FORWARD (2023-2025).
Benchmark: Buy & Hold and Binary Sentinel.
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
MIN_REGIME_BARS = 10 * 96

def get_signal_and_labels(prices):
    N = len(prices)
    # 200d SMA
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
    peak = 0
    mdd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > mdd: mdd = dd
    
    # Sharpe (approx daily)
    daily_eq = np.array(equity)[::96]
    daily_rets = np.diff(daily_eq) / daily_eq[:-1]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(365) if len(daily_rets) > 1 else 0
    
    return ret, cagr, mdd * 100, sharpe

# ============================================================================
# 1. RUN SIMULATION
# ============================================================================
print("Loading data...")
prices = np.load(NPY_PATH)
N = len(prices)
timestamps = [START_DT + datetime.timedelta(minutes=BAR_MIN * i) for i in range(N)]
idx_2019 = next(i for i, t in enumerate(timestamps) if t >= datetime.datetime(2019, 1, 1))
idx_2023 = next(i for i, t in enumerate(timestamps) if t >= datetime.datetime(2023, 1, 1))

signal, labels = get_signal_and_labels(prices)

def simulate_adaptive(p_subset, l_subset, s_subset):
    equity = [1.0]
    last_major = "DOWN"
    
    for i in range(1, len(l_subset)):
        lbl = l_subset[i]
        sig = s_subset[i]
        
        if lbl == "UP":
            # Adaptive scaling for UP state: Floor is 0.67, peaks at 1.0 (15% deviation)
            exposure = min(1.0, abs(sig) / (ENTER_PCT * 1.5))
            last_major = "UP"
        elif lbl == "DOWN":
            exposure = 0.0
            last_major = "DOWN"
        else: 
            # Memory-based scaling for structural participation
            exposure = 0.6 if last_major == "UP" else 0.2
            
        day_ret = 1.0 + exposure * ((p_subset[i] / p_subset[i-1]) - 1.0)
        equity.append(equity[-1] * day_ret)
        
    return np.array(equity)

def simulate_binary(p_subset, l_subset):
    equity = [1.0]
    for i in range(1, len(l_subset)):
        exposure = 1.0 if l_subset[i] == "UP" else 0.0
        day_ret = 1.0 + exposure * ((p_subset[i] / p_subset[i-1]) - 1.0)
        equity.append(equity[-1] * day_ret)
    return np.array(equity)

# Prepare Segments
f_p = prices[idx_2019:]
f_l = labels[idx_2019:]
f_s = signal[idx_2019:]
f_t = timestamps[idx_2019:]

# Simulation
eq_bh       = f_p / f_p[0]
eq_binary   = simulate_binary(f_p, f_l)
eq_adaptive = simulate_adaptive(f_p, f_l, f_s)

# ============================================================================
# 2. AUDIT OUTPUT
# ============================================================================
def print_period(name, e_full, split_idx):
    e_f = e_full
    e_w = e_full[split_idx:] / e_full[split_idx]
    
    f_r, f_c, f_m, f_s = calc_metrics(e_f)
    w_r, w_c, w_m, w_s = calc_metrics(e_w)
    
    print(f"\n--- {name} ---")
    print(f"FULL:    Ret {f_r:7.1f}% | CAGR {f_c:5.1f}% | MDD {f_m:4.1f}% | Sharpe {f_s:4.2f}")
    print(f"FORWARD: Ret {w_r:7.1f}% | CAGR {w_c:5.1f}% | MDD {w_m:4.1f}% | Sharpe {w_s:4.2f}")

split_off = idx_2023 - idx_2019
print_period("BUY & HOLD", eq_bh, split_off)
print_period("SENTINEL (BINARY 100/0)", eq_binary, split_off)
print_period("SENTINEL (ADAPTIVE EXPOSURE)", eq_adaptive, split_off)

# Plotting
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(f_t, eq_bh, color="#546e7a", alpha=0.4, label="Buy & Hold", lw=1)
ax.plot(f_t, eq_binary, color="#00e676", label="Binary Sentinel", lw=1.2)
ax.plot(f_t, eq_adaptive, color="#ffeb3b", label="Adaptive Sentinel (Smarter Scaling)", lw=2)

ax.axvline(datetime.datetime(2023, 1, 1), color="white", ls="--", alpha=0.3)
ax.set_yscale('log')
ax.set_title("Adaptive Exposure Sentinel: Strength-Based + Memory Scaling (2019-2025)", fontsize=14)
ax.legend(facecolor="#102027", edgecolor="#37474f")
ax.grid(True, which='both', color='#263238', lw=0.5)
plt.tight_layout()
plt.savefig("adaptive_exposure_performance.png", dpi=160)
print("\nResults visualised in adaptive_exposure_performance.png")
