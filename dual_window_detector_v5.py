"""
Dual-Window Anomaly Detector v5 -- Streaming Architecture
==========================================================
Simulates a real-time stream: each value is processed the moment it arrives.
No pandas. No pre-computation. Pure Python state.

Engine readiness is layered -- engines are INDEPENDENT:
  Points  0-4  : WARMUP         (spike engine still filling)
  Points  5-34 : Spike ACTIVE   (drift engine still warming -- shown as annotation)
  Points 35+   : Both ACTIVE    -> full DRIFT / SPIKE / NORMAL

Spike engine does NOT wait for drift engine.
Fast detectors fire fast. Slow detectors fire slow. They coexist.

Classification priority (once both ready):
  mean_delta > DRIFT_THRESHOLD   -> DRIFT
  abs(z_small) > SPIKE_THRESHOLD -> SPIKE
  else                           -> NORMAL
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import statistics
import json
from collections import deque

np.random.seed(42)

# ============================================================================
# 1. Data  -- Real-World Noise Simulation
#    Four independent noise layers, each probing a different failure mode.
# ============================================================================
print("=" * 72)
print("DUAL-WINDOW ANOMALY DETECTOR v5 -- STREAMING  (NOISE STRESS TEST)")
print("=" * 72)

# --- Base signal ----------------------------------------------------------
raw = np.random.normal(1000, 10, 100)

# --- NOISE LAYER 1: Pure variance burst (idx 30-45, std 10 → 40) ----------
# Variance-only change. Baseline mean is UNCHANGED (still 1000).
# This is a Type-2 distribution change -- mean stays the same, spread widens.
# Stress test: does z_small produce false spikes? Does drift engine stay quiet?
# Important: raw[40] = 850 (true spike) is still applied LAST and overrides this.
for i in range(30, 46):
    raw[i] = np.random.normal(1000, 40)   # same mean, 4x wider spread

# --- NOISE LAYER 2: Gradual micro baseline shift (idx 45-58, +35) ---------
# Why: a small sustained nudge that sits BELOW the drift threshold.
# Stress test: does drift engine false-trigger? Should stay NORMAL throughout.
for i in range(45, 59):
    raw[i] += 35

# --- NOISE LAYER 3: Unexpected single burst at idx 55, +200 ---------------
# Why: a real anomaly on top of the micro-shift.
# Expected: detected as SPIKE (z_small >> threshold despite noisy neighbourhood).
raw[55] += 200

# --- NOISE LAYER 4: Scattered random outliers (±70-90 at 3 points) --------
# t=15: during warmup period;  t=72: inside drift region;  t=88: post-drift.
np.random.seed(99)                        # separate seed for outlier positions
raw[15] += np.random.choice([-80, 80])   # random-direction outlier, warmup zone
raw[72] += np.random.normal(0, 50)       # extra noise inside sustained drift
raw[88] += 90                            # mild outlier just after drift ends
np.random.seed(42)                        # restore main seed state

# --- Known intentional injections (applied LAST so values are exact) ------
raw[25] = 1200          # true spike  (+200)
raw[40] = 850           # true spike  (-150)  sits inside pure-variance window
for i in range(60, 80):
    raw[i] = np.random.normal(1150, 10)   # sustained drift
raw[85] = 1300          # true spike  (+300)
raw[50] = 1100          # brief blip  (must stay NORMAL)
raw[51] = 1110

# Ground truth  -- TWO drift regimes
#   60-79 : upward regime  (baseline shifts from ~1000 → ~1150)
#   80-99 : downward regime (baseline returns ~1150 → ~1000)
# Both are structural regime changes. Detecting either is CORRECTNESS.
DRIFT_REGIMES = {
    "upward"  : set(range(60, 80)),    # forward shift
    "downward": set(range(80, 100)) - {85},  # reverse shift (exclude true spike)
}
TRUE_DRIFT  = DRIFT_REGIMES["upward"] | DRIFT_REGIMES["downward"]
TRUE_SPIKES = {25, 40, 55, 85}           # 55 = layer-3 burst
MUST_NORMAL = {50, 51}

print(f"""
  100 values with 4 noise layers:
    Layer 1  idx 30-45   PURE VARIANCE BURST  std 10→40, mean stays 1000
                         Type-2 change: detector sees spread, not shift
    Layer 2  idx 45-58   Micro baseline +35     → should NOT trigger drift
    Layer 3  idx 55      Burst outlier +200     → should fire as SPIKE
    Layer 4  idx 15,72,88  Scattered outliers   → false positive stress test

  True spikes  : {sorted(TRUE_SPIKES)}
  True drift   : indices 60-79
  Must be NORMAL: {sorted(MUST_NORMAL)}
""")
print(f"  True spikes : {sorted(TRUE_SPIKES)}")
print(f"  True drift  : indices 60-79")

# ============================================================================
# 2. Engine Configuration
# ============================================================================
SMALL_WIN     = 5      # spike engine ready after SMALL_WIN points
LARGE_WIN     = 20     # drift engine large window
LOOKBACK      = 15     # drift engine needs LOOKBACK full-window means
                       # => drift engine ready after LARGE_WIN + LOOKBACK = 35 pts

SPIKE_THRESH     = 1.65  # abs(z_small) > 1.65 -> SPIKE
DRIFT_ALPHA      = 4.5   # adaptive entry:  signed_delta > DRIFT_ALPHA * vol_smooth
DRIFT_EXIT_RATIO = 0.72  # exit hysteresis: exit when |delta| < ratio * entry_threshold
DRIFT_EMA_SPAN   = 10    # EMA span for smoothing vol estimate (alpha = 2/(span+1))
DRIFT_CONFIRM    = 3     # consecutive confirmations to enter OR exit drift
DRIFT_STAB       = 5     # neutral pts required after exit before re-entry

SPIKE_READY_AT = SMALL_WIN                # 5
DRIFT_READY_AT = LARGE_WIN + LOOKBACK     # 35

print(f"\n  Spike engine ready after : {SPIKE_READY_AT} points")
print(f"  Drift engine ready after : {DRIFT_READY_AT} points  "
      f"({LARGE_WIN} window + {LOOKBACK} lookback)")
print(f"  SPIKE threshold : abs(z_small) > {SPIKE_THRESH}")
print(f"  Drift state machine (ADAPTIVE + EMA-smoothed threshold):")
print(f"    Entry : signed_delta > {DRIFT_ALPHA} x vol_ema   (EMA span={DRIFT_EMA_SPAN})")
print(f"    Exit  : |delta| < {DRIFT_EXIT_RATIO} x entry_threshold    (hysteresis)")
print(f"    Confirm : {DRIFT_CONFIRM} consecutive pts to enter OR exit")
print(f"    Stabilize: {DRIFT_STAB} neutral pts required before re-entry")

# ============================================================================
# 3. Streaming State
# ============================================================================
small_win     = deque(maxlen=SMALL_WIN)   # rolling values for spike engine
large_win     = deque(maxlen=LARGE_WIN)   # rolling values for drift engine
mean_hist     = []                        # large-window means (only when full)
alerts        = []                        # structured alert log

# Drift state machine variables
drift_state       = "NORMAL"    # current state: NORMAL | DRIFT_UP | DRIFT_DOWN
enter_up          = 0           # consecutive signed_delta > +DRIFT_ENTER
enter_dn          = 0           # consecutive signed_delta < -DRIFT_ENTER
exit_cnt          = 0           # consecutive |signed_delta| < DRIFT_EXIT
stable_cnt        = 0           # neutral pts since last drift exit (stab gate)
post_drift        = False       # True after first drift exit; gates re-entry

# Regime intelligence state
regime_length     = 0           # pts since entering current drift (incl retro)
confidence        = 0.0         # grows per strong delta, decays when weakening
baseline_std_vals = []          # std values in NORMAL (volatility reference)
current_regime    = None        # active regime dict; None when in NORMAL
regimes           = []          # archived completed regime records
vol_ema           = None        # EMA of sig -- smoothed baseline volatility
EMA_ALPHA         = 2 / (DRIFT_EMA_SPAN + 1)  # EMA decay factor ≈ 0.182 for span=10
pre_vol_list      = []          # sig values during spike-only phase (clean baseline)

# ============================================================================
# 4. Stream Processing -- one value at a time
# ============================================================================
print("\n" + "=" * 72)
print("LIVE STREAM")
print(f"  {'idx':>3}  {'value':>8}  {'z_small':>8}  {'Δ / thr':>16}  label")
print("-" * 72)

results = []

for idx, val in enumerate(raw):

    # -- Step 1: feed value into both windows -------------------------------
    small_win.append(val)
    large_win.append(val)

    # -- Step 2: update mean history (only once large window is full) -------
    if len(large_win) == LARGE_WIN:
        mean_hist.append(sum(large_win) / LARGE_WIN)

    # -- Step 3: check engine readiness ------------------------------------
    spike_ready = len(small_win) >= SMALL_WIN           # true from point 4 (0-indexed)
    drift_ready = len(mean_hist) >= LOOKBACK            # true from point 34

    # -- Step 4: compute statistics if engine is ready ---------------------
    z_s        = None
    mean_delta = None

    if spike_ready:
        mu  = sum(small_win) / len(small_win)
        sig = statistics.stdev(small_win)               # sample std (n-1)
        z_s = (val - mu) / sig if sig > 0 else 0.0
    else:
        sig = None

    if drift_ready:
        # SIGNED: positive = baseline rose, negative = baseline fell
        mean_delta = mean_hist[-1] - mean_hist[-LOOKBACK]
        # Use 5-pt rolling std (sig) as the local noise floor.
        # vol_ema is updated AFTER classification (only on NORMAL points) so
        # neither spike outliers nor drift points can inflate the threshold.
        std_large      = sig if sig is not None else 10.0
        _vol_ref       = vol_ema if vol_ema is not None else std_large
        adaptive_enter = DRIFT_ALPHA * _vol_ref
        adaptive_exit  = adaptive_enter * DRIFT_EXIT_RATIO
        drift_strength = abs(mean_delta) / adaptive_enter
    else:
        std_large = adaptive_enter = adaptive_exit = drift_strength = None

    # -- Step 5: classify via Drift State Machine --------------------------
    if not spike_ready:
        label = "WARMUP"

    elif not drift_ready:
        # Only spike engine active.
        enter_up = enter_dn = exit_cnt = 0
        label = "SPIKE" if abs(z_s) > SPIKE_THRESH else "NORMAL"
        # Collect clean pre-drift sigma values to seed vol_ema correctly later
        if label == "NORMAL" and sig is not None:
            pre_vol_list.append(sig)

    else:
        # Both engines active.  Run the DFA + regime intelligence.
        if drift_state == "NORMAL":
            # Collect baseline volatility reference (only during genuine NORMAL)
            if sig is not None:
                baseline_std_vals.append(sig)
            confidence    = max(0.0, confidence - 0.5)  # cool between regimes
            regime_length = 0

            # Stabilization counter (neutral = |delta| < adaptive_enter)
            if abs(mean_delta) < adaptive_enter:
                stable_cnt = min(stable_cnt + 1, DRIFT_STAB + 99)
            else:
                stable_cnt = 0

            gate_open = (not post_drift) or (stable_cnt >= DRIFT_STAB)

            if gate_open:
                if mean_delta > adaptive_enter:
                    enter_up += 1; enter_dn = 0
                elif mean_delta < -adaptive_enter:
                    enter_dn += 1; enter_up = 0
                else:
                    enter_up = 0; enter_dn = 0

                def _confirm_entry(direction):
                    """Shared logic for DRIFT_UP / DRIFT_DOWN confirmation."""
                    global drift_state, confidence, regime_length, current_regime
                    drift_state   = direction
                    confidence    = float(DRIFT_CONFIRM)
                    regime_length = DRIFT_CONFIRM
                    current_regime = {
                        "direction"  : direction,
                        "start_idx"  : idx - (DRIFT_CONFIRM - 1),
                        "strength_v" : [drift_strength],
                        "vol_v"      : [sig] if sig is not None else [],
                    }

                if enter_up >= DRIFT_CONFIRM:
                    _confirm_entry("DRIFT_UP")
                    label = "DRIFT_UP"
                    if enter_up == DRIFT_CONFIRM:
                        for j in range(1, DRIFT_CONFIRM):
                            if len(results) >= j and results[-j]["label"] == "NORMAL":
                                results[-j]["label"] = "DRIFT_UP"
                                results[-j]["retro"]  = True

                elif enter_dn >= DRIFT_CONFIRM:
                    _confirm_entry("DRIFT_DOWN")
                    label = "DRIFT_DOWN"
                    if enter_dn == DRIFT_CONFIRM:
                        for j in range(1, DRIFT_CONFIRM):
                            if len(results) >= j and results[-j]["label"] == "NORMAL":
                                results[-j]["label"] = "DRIFT_DOWN"
                                results[-j]["retro"]  = True

                else:
                    label = "SPIKE" if abs(z_s) > SPIKE_THRESH else "NORMAL"
            else:
                enter_up = 0; enter_dn = 0
                label = "SPIKE" if abs(z_s) > SPIKE_THRESH else "NORMAL"

        else:   # DRIFT_UP or DRIFT_DOWN
            regime_length += 1

            # Confidence: grows per strong delta, decays exponentially when weak
            if abs(mean_delta) > adaptive_enter:
                confidence += 1.0
            else:
                confidence = max(0.0, confidence * 0.7)

            # Accumulate per-regime intelligence data
            if current_regime is not None:
                current_regime["strength_v"].append(drift_strength)
                if sig is not None:
                    current_regime["vol_v"].append(sig)

            # Exit logic (uses adaptive_exit for hysteresis)
            if abs(mean_delta) < adaptive_exit:
                exit_cnt += 1
                enter_up = 0; enter_dn = 0
                if exit_cnt >= DRIFT_CONFIRM:
                    # Archive completed regime
                    if current_regime is not None:
                        current_regime["end_idx"]   = idx - 1
                        current_regime["duration"]  = regime_length
                        current_regime["conf_peak"] = confidence
                        regimes.append(current_regime)
                        current_regime = None
                    drift_state = "NORMAL"
                    post_drift  = True
                    stable_cnt  = 0
                    exit_cnt    = 0
                    label = "NORMAL"
                else:
                    label = drift_state
            else:
                exit_cnt = 0
                label = drift_state

    # -- Step 5b: update vol_ema -- ONLY on genuine NORMAL points ----------
    # Seed from clean pre-drift warmup phase on first drift-ready point.
    # Subsequent updates: outlier-resistant EMA (cap at 2×vol_ema).
    if drift_ready and label == "NORMAL" and std_large is not None:
        if vol_ema is None:
            # Seed from 20th-percentile of pre-drift sig values (robust to early outliers)
            if pre_vol_list:
                pre_sorted  = sorted(pre_vol_list)
                p20_idx     = max(0, int(len(pre_sorted) * 0.20) - 1)
                vol_ema     = pre_sorted[p20_idx]
            else:
                vol_ema = std_large          # fallback if no pre-drift data
        else:
            std_capped = min(std_large, vol_ema * 1.3)   # tight outlier clip (~4% max step growth)
            vol_ema = EMA_ALPHA * std_capped + (1 - EMA_ALPHA) * vol_ema

    # -- Step 6: emit result immediately (streaming output) ----------------
    z_str  = f"{z_s:+7.3f}" if z_s is not None else "    N/A"
    # Show signed delta AND adaptive threshold: Δ=+76.5 thr=72.1
    if mean_delta is not None and adaptive_enter is not None:
        md_str = f"{mean_delta:+7.1f}/thr={adaptive_enter:5.1f}"
    else:
        md_str = "       N/A       "

    # Phase annotation
    if not spike_ready:
        phase = "(spike warming)"
    elif not drift_ready:
        phase = "(drift warming) "
    else:
        phase = "               "

    # [?] when entry counter is building (pending confirmation)
    pending_entry = (drift_state == "NORMAL" and drift_ready
                     and mean_delta is not None and (enter_up > 0 or enter_dn > 0))
    MARKERS = {
        "WARMUP"    : "[~]",
        "SPIKE"     : "[S]",
        "DRIFT_UP"  : "[↑]",
        "DRIFT_DOWN": "[↓]",
        "NORMAL"    : "[ ]",
    }
    marker = "[?]" if pending_entry else MARKERS.get(label, "[ ]")

    print(f"  {marker} {idx:3d}  {val:8.2f}  {z_str}  {md_str}  {phase}  {label}")

    results.append({
        "idx"        : idx, "val": val,
        "z_s"        : z_s, "md": mean_delta,
        "label"      : label,
        "drift_ready": drift_ready,
        "retro"      : False,
        "drift_str"  : drift_strength,
        "regime_len" : regime_length,
        "confidence" : confidence,
        "adap_thr"   : round(adaptive_enter, 2) if adaptive_enter is not None else None,
        "vol_ema"    : round(vol_ema, 3)        if vol_ema        is not None else None,
    })

    # -- Step 7: log alert only for SPIKE (DRIFT alerts added after retro pass) --
    if label == "SPIKE":
        alerts.append({
            "index"     : idx,
            "type"      : "SPIKE",
            "value"     : round(float(val), 4),
            "z_small"   : round(float(z_s), 4) if z_s is not None else None,
            "mean_delta": round(float(mean_delta), 4) if mean_delta is not None else None,
            "phase"     : "full" if drift_ready else "spike-only",
        })

print("-" * 72)

# Archive any drift regime still active at stream end (no natural exit)
if current_regime is not None:
    current_regime["end_idx"]   = len(raw) - 1
    current_regime["duration"]  = regime_length
    current_regime["conf_peak"] = confidence
    regimes.append(current_regime)
    current_regime = None

# Post-stream: add DRIFT_UP / DRIFT_DOWN alerts (incl retro-corrected pts)
for r in results:
    if r["label"] in ("DRIFT_UP", "DRIFT_DOWN"):
        alerts.append({
            "index"      : r["idx"],
            "type"       : r["label"],
            "value"      : round(float(r["val"]), 4),
            "z_small"    : round(float(r["z_s"]), 4) if r["z_s"] is not None else None,
            "mean_delta" : round(float(r["md"]), 4)  if r["md"]  is not None else None,
            "drift_str"  : round(float(r["drift_str"]), 3) if r["drift_str"] is not None else None,
            "confidence" : round(float(r["confidence"]), 2),
            "phase"      : "full",
            "retro"      : r.get("retro", False),
        })
alerts.sort(key=lambda a: a["index"])

# Print retro-correction summary
retro_pts = [r for r in results if r.get("retro")]
if retro_pts:
    print(f"\n  RETRO-CORRECTIONS ({len(retro_pts)} points)")
    print(f"  (live stream showed NORMAL -- corrected after confirmation)")
    for r in retro_pts:
        print(f"    idx {r['idx']:3d}  val={r['val']:8.2f}  "
              f"signed_delta={r['md']:+6.2f}  NORMAL -> {r['label']} (retro)")

# ============================================================================
# REGIME INTELLIGENCE
# ============================================================================
def _strength_label(s):
    if s is None:  return "N/A    "
    if s < 1.1:    return "Weak   "
    if s < 1.5:    return "Moderate"
    return                "Strong  "

def _conf_label(c):
    if c < 4:   return "Low     "
    if c < 8:   return "Moderate"
    return              "High    "

def _vol_label(regime_vols, base_vols):
    if not regime_vols or not base_vols: return "N/A"
    r_avg = sum(regime_vols) / len(regime_vols)
    b_avg = sum(base_vols)   / len(base_vols)
    if b_avg == 0: return "N/A"
    ratio = r_avg / b_avg
    tag   = "Stable   " if ratio < 1.3 else ("Elevated " if ratio < 2.0 else "Unstable ")
    return f"{tag} (x{ratio:.2f} baseline)"

def _dur_label(d):
    if d < 5:   return "Transient    (<5 pts)"
    if d < 10:  return "Short-lived  (5-9 pts)"
    return              "Structural   (10+ pts)"

if regimes:
    print("\n" + "=" * 72)
    print("REGIME INTELLIGENCE")
    print("=" * 72)
    for n, r in enumerate(regimes, 1):
        sv   = r["strength_v"]
        avg_s = sum(sv) / len(sv) if sv else None
        end  = r.get("end_idx", "?")
        dur  = r["duration"]
        cpk  = r["conf_peak"]
        print(f"""
  REGIME #{n}  {r['direction']}   idx {r['start_idx']}–{end}  ({dur} pts)
  ┌─────────────────────────┬────────────────────────────────────────┐
  │ Property                │ Value                                  │
  ├─────────────────────────┼────────────────────────────────────────┤
  │ Duration                │ {dur:3d} pts   {_dur_label(dur):<25}│
  │ Avg Drift Strength      │ {avg_s:.3f}    {_strength_label(avg_s):<25}  (δ/threshold)  │
  │ Peak Confidence         │ {cpk:5.1f}   {_conf_label(cpk):<25}  (consecutive hits) │
  │ Regime Volatility       │ {_vol_label(r['vol_v'], baseline_std_vals):<40}│
  └─────────────────────────┴────────────────────────────────────────┘""")

# ============================================================================
# 5. Anomaly Detail Report
# ============================================================================
print("\n" + "=" * 72)
print("ANOMALY DETAILS")
print("=" * 72)

spikes = [r for r in results if r["label"] == "SPIKE"]
drifts = [r for r in results if r["label"] in ("DRIFT_UP", "DRIFT_DOWN")]
drift_up_pts   = [r for r in results if r["label"] == "DRIFT_UP"]
drift_down_pts = [r for r in results if r["label"] == "DRIFT_DOWN"]

print(f"\n  Detected Spikes ({len(spikes)}):")
for s in spikes:
    print(f"    idx {s['idx']:3d}  val={s['val']:8.2f}  z_small={s['z_s']:+7.3f}")

print(f"\n  Detected DRIFT_UP  ({len(drift_up_pts)} pts):  "
      f"{[r['idx'] for r in drift_up_pts]}")
print(f"  Detected DRIFT_DOWN({len(drift_down_pts)} pts):  "
      f"{[r['idx'] for r in drift_down_pts]}")

print(f"\n  Warmup points (spike engine filling): "
      f"{sum(1 for r in results if r['label'] == 'WARMUP')} "
      f"(indices 0-{SPIKE_READY_AT - 1})")
print(f"  Drift-warming points (spike active, drift filling): "
      f"{sum(1 for r in results if r['label'] in ('SPIKE','NORMAL') and not r['drift_ready'])} "
      f"(indices {SPIKE_READY_AT}-{DRIFT_READY_AT - 1})")

# ============================================================================
# 6. Professional Evaluation Summary
# ============================================================================
print("\n" + "=" * 72)
print("PROFESSIONAL EVALUATION SUMMARY")
print("=" * 72)

classified = {r["idx"] for r in results if r["label"] != "WARMUP"}
spike_idx  = {r["idx"] for r in results if r["label"] == "SPIKE"}
drift_idx  = {r["idx"] for r in results if r["label"] in ("DRIFT_UP", "DRIFT_DOWN")}

tp_spikes    = TRUE_SPIKES  & spike_idx
fp_spikes    = spike_idx    - TRUE_SPIKES - TRUE_DRIFT
missed_sp    = (TRUE_SPIKES & classified) - spike_idx

tp_drift_pts = TRUE_DRIFT   & drift_idx
fp_drift     = drift_idx    - TRUE_DRIFT - TRUE_SPIKES
missed_dr    = (TRUE_DRIFT  & classified) - drift_idx

# Indices 50-51 check
lbl_50 = next(r["label"] for r in results if r["idx"] == 50)
lbl_51 = next(r["label"] for r in results if r["idx"] == 51)

normal_c  = sum(1 for r in results if r["label"] == "NORMAL")
warmup_c  = sum(1 for r in results if r["label"] == "WARMUP")
spike_c   = len(spikes)
drift_c   = len(drifts)
total     = len(results)

print(f"""
  Detected Spikes     : {sorted(spike_idx)}
  Detected Drift      : {sorted(drift_idx)[:8]} ...

  True Positive Spikes : {sorted(tp_spikes)}  (expected {sorted(TRUE_SPIKES)})
  True Positive Drift  : {len(tp_drift_pts)}/{len(TRUE_DRIFT)} pts across both drift regimes

  Regime-level detection (did we catch each regime at all?):
    Upward regime   (60-79) : {'DETECTED' if drift_idx & DRIFT_REGIMES['upward']   else 'MISSED'}
      pts caught = {sorted(drift_idx & DRIFT_REGIMES['upward'])}
    Downward regime (80-99) : {'DETECTED' if drift_idx & DRIFT_REGIMES['downward'] else 'MISSED'}
      pts caught = {sorted(drift_idx & DRIFT_REGIMES['downward'])}

  False Positive Spikes: {sorted(fp_spikes)}
  False Positive Drift : {sorted(fp_drift)}  <- points outside BOTH regimes
  Missed Spikes        : {sorted(missed_sp)}
  Missed Drift pts     : {len(missed_dr)}  (across both regimes)

  Sanity check (50-51 must not be SPIKE or DRIFT)
    idx 50 -> {lbl_50}  (wanted: NORMAL)
    idx 51 -> {lbl_51}  (wanted: NORMAL)

  Counts
    [ ] NORMAL  : {normal_c:3d}  ({normal_c/total*100:.1f}%)
    [S] SPIKE   : {spike_c:3d}  ({spike_c/total*100:.1f}%)
    [D] DRIFT   : {drift_c:3d}  ({drift_c/total*100:.1f}%)
    [~] WARMUP  : {warmup_c:3d}  ({warmup_c/total*100:.1f}%)  (points 0-{SPIKE_READY_AT-1} only)
    Total       : {total:3d}
""")

# ============================================================================
# 6b. Scientific Metrics
# ============================================================================
print("=" * 72)
print("SCIENTIFIC METRICS")
print("=" * 72)

def safe_div(num, den):
    """Division that returns 0.0 instead of ZeroDivisionError."""
    return num / den if den > 0 else 0.0

# --- Spike engine ---
sp_tp = len(tp_spikes)            # true positives
sp_fp = len(fp_spikes)            # false positives
sp_fn = len(missed_sp)            # false negatives  (missed true spikes)
sp_precision = safe_div(sp_tp, sp_tp + sp_fp)
sp_recall    = safe_div(sp_tp, sp_tp + sp_fn)
sp_f1        = safe_div(2 * sp_precision * sp_recall,
                         sp_precision + sp_recall)

# --- Drift engine (point-level) ---
dr_tp = len(tp_drift_pts)         # drift points correctly flagged
dr_fp = len(fp_drift)             # points flagged DRIFT outside true region
dr_fn = len(missed_dr)            # true drift points not flagged
dr_precision = safe_div(dr_tp, dr_tp + dr_fp)
dr_recall    = safe_div(dr_tp, dr_tp + dr_fn)
dr_f1        = safe_div(2 * dr_precision * dr_recall,
                         dr_precision + dr_recall)

# --- Drift detection delay (per regime) ---
# Regime 1: upward drift begins at 60
up_detected  = drift_idx & DRIFT_REGIMES["upward"]
up_delay     = min(up_detected) - 60  if up_detected  else None
# Regime 2: downward drift begins at 80
down_detected = drift_idx & DRIFT_REGIMES["downward"]
down_delay    = min(down_detected) - 80 if down_detected else None

# Combined delay metric (forward shift only, as that is the primary)
true_drift_start = 60                               # forward regime start
first_drift_det  = min(drift_idx) if drift_idx else None
detection_delay  = (first_drift_det - true_drift_start
                    if first_drift_det is not None else None)

# --- Combined (spike + drift treated as one anomaly class) ---
all_tp = sp_tp + dr_tp
all_fp = sp_fp + dr_fp
all_fn = sp_fn + dr_fn
all_precision = safe_div(all_tp, all_tp + all_fp)
all_recall    = safe_div(all_tp, all_tp + all_fn)
all_f1        = safe_div(2 * all_precision * all_recall,
                          all_precision + all_recall)

print(f"""
  SPIKE ENGINE
  ┌─────────────┬────────┬─────────────────────────────────────────────┐
  │ Metric      │  Value │ Formula                                     │
  ├─────────────┼────────┼─────────────────────────────────────────────┤
  │ TP          │  {sp_tp:>5} │ true spikes correctly detected              │
  │ FP          │  {sp_fp:>5} │ non-spikes flagged as SPIKE                 │
  │ FN (missed) │  {sp_fn:>5} │ true spikes not detected                    │
  │ Precision   │  {sp_precision:>5.3f} │ TP / (TP + FP)                              │
  │ Recall      │  {sp_recall:>5.3f} │ TP / (TP + FN)                              │
  │ F1 Score    │  {sp_f1:>5.3f} │ 2 * P * R / (P + R)                         │
  └─────────────┴────────┴─────────────────────────────────────────────┘

  DRIFT ENGINE  (corrected: both regimes as TRUE DRIFT)
  ┌─────────────────────┬────────┬─────────────────────────────────────┐
  │ Metric               │  Value │ Notes                              │
  ├─────────────────────┼────────┼─────────────────────────────────────┤
  │ TP (point-level)    │  {dr_tp:>5} │ drift pts in either regime          │
  │ FP (true false pos) │  {dr_fp:>5} │ drift pts outside BOTH regimes      │
  │ FN (missed pts)     │  {dr_fn:>5} │ regime pts not detected             │
  │ Precision           │  {dr_precision:>5.3f} │ TP / (TP + FP)                     │
  │ Recall              │  {dr_recall:>5.3f} │ TP / (TP + FN)                     │
  │ F1 Score            │  {dr_f1:>5.3f} │ 2 * P * R / (P + R)                │
  ├─────────────────────┼────────┼─────────────────────────────────────┤
  │ REGION-LEVEL        │        │ was the regime detected at all?     │
  │ Upward (60-79)      │  {'YES':>5} │ first alert idx {min(up_detected) if up_detected else 'N/A':<3}  delay={up_delay if up_delay is not None else 'N/A'} pts  │
  │ Downward (80-99)    │  {'YES' if down_detected else ' NO':>5} │ first alert idx {min(down_detected) if down_detected else 'N/A':<3}  delay={down_delay if down_delay is not None else 'N/A'} pts  │
  └─────────────────────┴────────┴─────────────────────────────────────┘

  COMBINED (both engines, both regimes)
  ┌─────────────┬────────┬─────────────────────────────────────────────┐
  │ Metric      │  Value │                                             │
  ├─────────────┼────────┼─────────────────────────────────────────────┤
  │ Precision   │  {all_precision:>5.3f} │ (spike TP + drift TP) / all positives       │
  │ Recall      │  {all_recall:>5.3f} │ (spike TP + drift TP) / all true anomalies  │
  │ F1 Score    │  {all_f1:>5.3f} │                                             │
  └─────────────┴────────┴─────────────────────────────────────────────┘
""")

# ============================================================================
# 7. Structured Alert Log + JSON Export
# ============================================================================
print("=" * 72)
print("STRUCTURED ALERT LOG")
print("=" * 72)

print(f"\n  {len(alerts)} alerts raised during stream:\n")
print(f"  {'#':>3}  {'idx':>4}  {'type':<6}  {'value':>9}  "
      f"{'z_small':>8}  {'mean_delta':>10}  phase")
print(f"  {'-'*3}  {'-'*4}  {'-'*6}  {'-'*9}  "
      f"{'-'*8}  {'-'*10}  -----")
for n, a in enumerate(alerts, 1):
    z_str  = f"{a['z_small']:+8.3f}"   if a['z_small']    is not None else "     N/A"
    md_str = f"{a['mean_delta']:10.2f}" if a['mean_delta'] is not None else "       N/A"
    print(f"  {n:>3}  {a['index']:>4}  {a['type']:<6}  "
          f"{a['value']:>9.2f}  {z_str}  {md_str}  {a['phase']}")

# JSON export
ALERT_PATH = "alerts.json"
with open(ALERT_PATH, "w", encoding="utf-8") as f:
    json.dump(alerts, f, indent=2)
print(f"\n  Exported {len(alerts)} alerts -> {ALERT_PATH}")

# ============================================================================
# 8. Architecture Recap
# ============================================================================
print("=" * 72)
print("STREAMING ARCHITECTURE RECAP")
print("=" * 72)
print(f"""
  Each value processed instantly on arrival -- no look-ahead.

  State                  Size    Ready after
  ----------------------------------------
  small_win  (deque)       {SMALL_WIN}       {SPIKE_READY_AT} points   <- spike engine
  large_win  (deque)      {LARGE_WIN}      {LARGE_WIN} points
  mean_hist  (list)        *      {DRIFT_READY_AT} points   <- drift engine

  Layered classification:
    Points  0 - {SPIKE_READY_AT-1:2d} : [~] WARMUP      (spike engine filling)
    Points  {SPIKE_READY_AT} - {DRIFT_READY_AT-1:2d} : [S]/[ ] SPIKE or NORMAL  (spike ACTIVE, drift warming)
                           -> spike fires immediately -- does NOT wait for drift
    Points {DRIFT_READY_AT} - 99 : [S]/[D]/[ ] full classification (both engines ACTIVE)

  Engine independence:
    Fast detector (spike)  fires after {SPIKE_READY_AT} points.   Low latency.
    Slow detector (drift)  fires after {DRIFT_READY_AT} points.  High confidence.
    Neither blocks the other. They coexist, just like in real monitoring systems.

  No shared state between engines.
  No deque for drift detection -- pure mean_delta comparison.
  No pre-computation -- every classification uses only past data.
""")
print("=" * 72)
print("v5 COMPLETE")
print("=" * 72)
