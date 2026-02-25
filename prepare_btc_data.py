"""
prepare_btc_data.py
===================
Validates and cleans btc_15m_data_2018_to_2025.csv for Sentinel drift detection.

Rules applied (as requested):
  1. Sort by timestamp ascending
  2. Remove exact duplicate timestamps (keep first)
  3. Detect missing 15-min bars; forward-fill gaps of ≤4 bars (≤60 min)
  4. Flag gaps > 4 bars for awareness (no fill — raw price preserved)
  5. Use Close price only
  6. No normalisation — raw USD values

Outputs:
  btc_close_clean.npy   numpy array of float64 Close prices (ready to stream)
  btc_gaps_report.csv   human-readable gap log
"""

import csv
import math
import datetime
import statistics
import os

INPUT_FILE  = r"C:\Users\yashw\Projects\Sentinel\btc_15m_data_2018_to_2025.csv"
OUT_NPY     = r"C:\Users\yashw\Projects\Sentinel\btc_close_clean.npy"
OUT_GAPS    = r"C:\Users\yashw\Projects\Sentinel\btc_gaps_report.csv"
BAR_SECONDS = 15 * 60          # 900 s per 15-min candle
MAX_FILL    = 4                 # forward-fill up to this many missing bars (<=60 min gap)

# ---------------------------------------------------------------------------
# 1. Load raw rows (timestamp + Close only)
# ---------------------------------------------------------------------------
print("Loading CSV …")
rows = []
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    # Locate columns by name (defensive)
    h = [c.strip() for c in header]
    ts_col    = h.index("Open time")
    close_col = h.index("Close")

    for i, row in enumerate(reader):
        try:
            ts_raw = row[ts_col].strip()
            # Binance timestamps look like "2018-01-01 00:00:00.000000"
            # Accept with or without microseconds
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    ts = datetime.datetime.strptime(ts_raw, fmt)
                    break
                except ValueError:
                    continue
            else:
                print(f"  [WARN] row {i+2}: unparseable timestamp '{ts_raw}' — skipped")
                continue
            close = float(row[close_col])
            rows.append((ts, close))
        except Exception as e:
            print(f"  [WARN] row {i+2}: {e} — skipped")

print(f"  Loaded {len(rows):,} rows")

# ---------------------------------------------------------------------------
# 2. Sort ascending by timestamp
# ---------------------------------------------------------------------------
rows.sort(key=lambda r: r[0])

# ---------------------------------------------------------------------------
# 3. Remove duplicate timestamps (keep first occurrence after sort)
# ---------------------------------------------------------------------------
seen   = set()
deduped = []
dup_count = 0
for ts, close in rows:
    key = ts
    if key in seen:
        dup_count += 1
    else:
        seen.add(key)
        deduped.append((ts, close))

print(f"  Duplicates removed : {dup_count:,}")
rows = deduped

# ---------------------------------------------------------------------------
# 4. Gap detection and forward-fill
# ---------------------------------------------------------------------------
gaps        = []      # (start_ts, end_ts, missing_bars, action)
filled      = 0
major_gaps  = 0
clean_rows  = [rows[0]]   # seed with first row

for i in range(1, len(rows)):
    prev_ts, prev_close = clean_rows[-1]
    cur_ts,  cur_close  = rows[i]
    expected_ts = prev_ts + datetime.timedelta(seconds=BAR_SECONDS)

    if cur_ts == expected_ts:
        # Consecutive bar — normal
        clean_rows.append(rows[i])
    elif cur_ts > expected_ts:
        # Gap detected
        missing = int(round((cur_ts - expected_ts).total_seconds() / BAR_SECONDS))
        action  = "forward-filled" if missing <= MAX_FILL else "left-as-gap (major)"
        gaps.append({
            "gap_start"   : expected_ts.strftime("%Y-%m-%d %H:%M"),
            "gap_end"     : cur_ts.strftime("%Y-%m-%d %H:%M"),
            "missing_bars": missing,
            "action"      : action,
            "fwd_price"   : round(prev_close, 2),
        })
        if missing <= MAX_FILL:
            # Forward-fill
            fill_ts = expected_ts
            for _ in range(missing):
                clean_rows.append((fill_ts, prev_close))
                fill_ts += datetime.timedelta(seconds=BAR_SECONDS)
                filled += 1
        else:
            major_gaps += 1
        clean_rows.append(rows[i])
    else:
        # cur_ts < expected_ts after dedup — shouldn't happen, skip
        print(f"  [WARN] out-of-order row at {cur_ts} after sort/dedup — skipped")

print(f"  Gaps found         : {len(gaps):,}  "
      f"({len(gaps)-major_gaps} minor filled, {major_gaps} major left)")
print(f"  Bars forward-filled: {filled:,}")
print(f"  Final bar count    : {len(clean_rows):,}")

# ---------------------------------------------------------------------------
# 5. Write gap report CSV
# ---------------------------------------------------------------------------
with open(OUT_GAPS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["gap_start","gap_end","missing_bars","action","fwd_price"])
    writer.writeheader()
    writer.writerows(gaps)
print(f"  Gap report written : {OUT_GAPS}")

# ---------------------------------------------------------------------------
# 6. Extract Close array and write .npy (pure python — no numpy dependency)
#    Falls back to writing a plain CSV if numpy not installed.
# ---------------------------------------------------------------------------
close_prices = [float(c) for _, c in clean_rows]

# Quick sanity stats
n     = len(close_prices)
mn    = min(close_prices)
mx    = max(close_prices)
mu    = sum(close_prices) / n
sigma = statistics.stdev(close_prices)

print(f"\n  CLOSE PRICE STATS")
print(f"    Bars       : {n:,}")
print(f"    Date range : {clean_rows[0][0]}  →  {clean_rows[-1][0]}")
print(f"    Min price  : ${mn:,.2f}")
print(f"    Max price  : ${mx:,.2f}")
print(f"    Mean       : ${mu:,.2f}")
print(f"    Std dev    : ${sigma:,.2f}")

try:
    import struct
    # Write as raw little-endian float64 binary (numpy-compatible)
    # numpy.load() on a raw binary won't work — use actual npy format
    # Build a minimal .npy v1.0 header manually
    import io
    dtype_str = b"<f8"   # little-endian float64
    shape_str = f"({n},)"
    # numpy magic + 0x01 0x00 (major, minor version)
    MAGIC = b"\x93NUMPY\x01\x00"
    header_dict = (
        f"{{'descr': '<f8', 'fortran_order': False, 'shape': ({n},), }}"
    ).encode("latin1")
    # Pad to multiple of 64
    pad = 64 - ((len(MAGIC) + 2 + len(header_dict) + 1) % 64)
    header_dict += b" " * pad + b"\n"
    header_len = len(header_dict).to_bytes(2, "little")
    with open(OUT_NPY, "wb") as f:
        f.write(MAGIC)
        f.write(header_len)
        f.write(header_dict)
        f.write(struct.pack(f"<{n}d", *close_prices))
    print(f"\n  Saved .npy         : {OUT_NPY}")
except Exception as e:
    # Fallback: plain CSV
    fallback = OUT_NPY.replace(".npy", ".csv")
    with open(fallback, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["close"])
        for p in close_prices:
            w.writerow([p])
    print(f"\n  [WARN] .npy write failed ({e}); saved CSV: {fallback}")

print("\n  Done. btc_close_clean.npy is ready for dual_window_detector_v5.py")
