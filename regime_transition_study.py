"""
regime_transition_study.py
==========================
Analyzes the structural dynamics of BTC market regimes (2018-2025).

Core Objectives:
1. Transition Probability Matrix: Quantify how regimes flip.
2. Regime Persistence: Measures average duration and hazard rates.
3. Autocorrelation: Test for structural regime memory.

Parameters (Frozen):
- SMA: 200 days (19,200 bars)
- Enter: 10% deviation
- Exit: 4% deviation
- Confirm: 3 bars
- Min Duration: 10 days (960 bars)
"""

import datetime
import statistics
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

# ============================================================================
# 0. CONFIG & DATA
# ============================================================================
NPY_PATH = r"C:\Users\yashw\Projects\Sentinel\btc_regime_analysis\btc_close_clean.npy"
BAR_MIN  = 15
START_DT = datetime.datetime(2018, 1, 1)

# Frozen Parameters
SMA_WIN         = 19_200 
ENTER_PCT       = 0.10   
EXIT_PCT        = 0.04   
DRIFT_CONFIRM   = 3      
MIN_REGIME_BARS = 10 * 96

def get_regime_labels(prices):
    N = len(prices)
    # 1. SMA Compute
    sma = np.full(N, np.nan)
    win_sum = 0.0
    for i in range(N):
        win_sum += prices[i]
        if i >= SMA_WIN: win_sum -= prices[i - SMA_WIN]
        if i >= SMA_WIN - 1: sma[i] = win_sum / SMA_WIN
    
    signal = np.where(np.isnan(sma), np.nan, (prices - sma) / sma)

    # 2. DFA (Drift State Machine)
    drift_state = "NORMAL"
    enter_up = enter_dn = exit_cnt = 0
    labels = []

    for sig in signal:
        if np.isnan(sig):
            labels.append("NORMAL")
            continue
        
        if drift_state == "NORMAL":
            if sig > ENTER_PCT:
                enter_up += 1; enter_dn = 0
            elif sig < -ENTER_PCT:
                enter_dn += 1; enter_up = 0
            else:
                enter_up = enter_dn = 0

            if enter_up >= DRIFT_CONFIRM:
                drift_state = "UP"; label = "UP"
                for j in range(1, DRIFT_CONFIRM):
                    if len(labels) >= j and labels[-j] == "NORMAL": labels[-j] = "UP"
            elif enter_dn >= DRIFT_CONFIRM:
                drift_state = "DOWN"; label = "DOWN"
                for j in range(1, DRIFT_CONFIRM):
                    if len(labels) >= j and labels[-j] == "NORMAL": labels[-j] = "DOWN"
            else:
                label = "NORMAL"
        
        elif drift_state == "UP":
            if sig < EXIT_PCT:
                exit_cnt += 1
                if exit_cnt >= DRIFT_CONFIRM:
                    drift_state = "NORMAL"; exit_cnt = enter_up = enter_dn = 0; label = "NORMAL"
                else: label = "UP"
            else: exit_cnt = 0; label = "UP"
            
        else: 
            if sig > -EXIT_PCT:
                exit_cnt += 1
                if exit_cnt >= DRIFT_CONFIRM:
                    drift_state = "NORMAL"; exit_cnt = enter_up = enter_dn = 0; label = "NORMAL"
                else: label = "DOWN"
            else: exit_cnt = 0; label = "DOWN"
        
        labels.append(label)

    # 3. Structural Cleanup
    segments = []
    if not labels: return [], []
    start_i = 0
    curr = labels[0]
    for i in range(1, N):
        if labels[i] != curr:
            segments.append([start_i, i-1, curr])
            start_i, curr = i, labels[i]
    segments.append([start_i, N-1, curr])

    
    changed = True
    while changed:
        changed = False
        out = []
        for i, seg in enumerate(segments):
            s, e, lbl = seg
            if (e - s + 1) < MIN_REGIME_BARS:
                if out:
                    out[-1][1] = e # Absorb left
                    changed = True
                elif i + 1 < len(segments):
                    segments[i+1][0] = s # Next segment absorbs this
                    changed = True
                else: out.append(seg)
            else:
                if out and out[-1][2] == lbl:
                    out[-1][1] = e
                    changed = True
                else: out.append(seg)
        segments = out

    # Re-expand labels
    final_labels = ["NORMAL"] * N
    for s, e, lbl in segments:
        for i in range(s, e+1): final_labels[i] = lbl
    
    return final_labels, segments

# ============================================================================
# 1. LOAD DATA & GENERATE LABELS
# ============================================================================
print("Loading BTC data...")
prices = np.load(NPY_PATH)
N = len(prices)
print(f"Dataset: {N:,} bars (15-min resolution)")

labels, segments = get_regime_labels(prices)

# ============================================================================
# 2. TRANSITION PROBABILITY MATRIX
# ============================================================================
print("\nComputing Transition Matrix...")
states = ["UP", "NORMAL", "DOWN"]
state_to_idx = {s: i for i, s in enumerate(states)}
counts = np.zeros((3, 3))

# We measure transitions at the segment level (macro transitions)
for i in range(1, len(segments)):
    prev_state = segments[i-1][2]
    curr_state = segments[i][2]
    counts[state_to_idx[prev_state], state_to_idx[curr_state]] += 1

# Normalize rows to get probabilities
# Note: P(i|i) is 0 here because we are looking at segment transitions.
# To get bar-level probabilities, we'd look at labels[i] -> labels[i+1].
# But structural study usually cares about "When we leave UP, where do we go?"

probs = counts / counts.sum(axis=1, keepdims=True)

# ============================================================================
# 3. REGIME PERSISTENCE
# ============================================================================
print("\nComputing Persistence Metrics...")
durations = {s: [] for s in states}
for s, e, lbl in segments:
    durations[lbl].append((e - s + 1) / 96) # convert to days

print(f"{'State':<10} | {'Count':<6} | {'Avg Dur (days)':<15} | {'Med Dur (days)':<15}")
print("-" * 55)
for s in states:
    d = durations[s]
    avg_d = sum(d) / len(d) if d else 0
    med_d = statistics.median(d) if d else 0
    print(f"{s:<10} | {len(d):<6} | {avg_d:15.2f} | {med_d:15.2f}")

# ============================================================================
# 4. AUTOCORRELATION OF REGIME STATES
# ============================================================================
print("\nComputing Regime Autocorrelation...")
# Encode: UP=1, NORMAL=0, DOWN=-1
encoding = {"UP": 1, "NORMAL": 0, "DOWN": -1}
encoded_series = np.array([encoding[l] for l in labels])

# Sampling to 1-day resolution to make ACF meaningful (lag 1 = 1 day)
daily_regimes = encoded_series[::96]
acf_vals = acf(daily_regimes, nlags=30)

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\nGenerating Study Report...")
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2)

# Subplot 1: Transition Heatmap
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(probs, annot=True, xticklabels=states, yticklabels=states, cmap="RdYlGn", ax=ax1, cbar=False)
ax1.set_title("Macro Transition Probabilities (Segment Level)")
ax1.set_ylabel("From")
ax1.set_xlabel("To")

# Subplot 2: Duration Distribution (Boxplot)
ax2 = fig.add_subplot(gs[0, 1])
boxplot_data = [durations[s] for s in states]
ax2.boxplot(boxplot_data, labels=states, patch_artist=True, 
            boxprops=dict(facecolor="#4fc3f7", alpha=0.5))
ax2.set_title("Regime Duration Distribution")
ax2.set_ylabel("Duration (Days)")

# Subplot 3: Autocorrelation Plot
ax3 = fig.add_subplot(gs[1, :])
ax3.stem(range(len(acf_vals)), acf_vals)
ax3.axhline(0, color="white", linestyle="--", alpha=0.5)
ax3.set_title("Regime State Autocorrelation (Daily Resolution)")
ax3.set_xlabel("Lag (Days)")
ax3.set_ylabel("ACF")
ax3.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig("btc_regime_analysis/regime_transition_study.png", dpi=150)
print("Analysis complete. Chart saved: btc_regime_analysis/regime_transition_study.png")

# Final Numbers Print
print("\n" + "="*40)
print("STRUCTURAL PERSISTENCE SUMMARY")
print("="*40)
print(f"Total Segments: {len(segments)}")
flip_up_to_down = counts[state_to_idx["UP"], state_to_idx["DOWN"]]
flip_down_to_up = counts[state_to_idx["DOWN"], state_to_idx["UP"]]
print(f"Direct UP -> DOWN Flips: {int(flip_up_to_down)}")
print(f"Direct DOWN -> UP Flips: {int(flip_down_to_up)}")
print(f"ACF Lag 1 (1 day persistence): {acf_vals[1]:.3f}")
print(f"ACF Lag 10 (10 day memory):    {acf_vals[10]:.3f}")
print(f"ACF Lag 30 (30 day memory):    {acf_vals[30]:.3f}")
print("="*40)
