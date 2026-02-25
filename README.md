# BTC Regime Analysis: The Sentinel Project 🛰️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance quantitative framework for detecting, analyzing, and trading Bitcoin market regimes. The **Sentinel Project** transcends static allocation models by employing a dual-window detection engine that adapts exposure based on market "drift" and structural memory.

---

## 🛡️ The Sentinel Framework

Sentinel is built on the premise that Bitcoin's market behavior is non-stationary. The system classifies market states into three distinct regimes, each triggering a unique tactical response:

| Regime | Market Behavior | Strategy Response |
| :--- | :--- | :--- |
| **`DRIFT_UP`** | Sustained bullish momentum | **Adaptive Exposure**: Scales from 60% to 100% based on trend strength. |
| **`NORMAL`** | Consolidation / Accumulation | **Memory-Based**: 60% exposure if emerging from UP; 20% if recovering from DOWN. |
| **`DRIFT_DOWN`** | Sustained bearish breakdown | **Capital Preservation**: 0% exposure (Zero tolerance for bear drifts). |

### 🧠 Tactical Intelligence
Unlike binary "on/off" models, Sentinel leverages two key indicators:
1.  **Trend Intensity**: Exposure is dynamically scaled based on the price deviation from the 200-period SMA.
2.  **State Persistence**: The strategy maintains "regime memory," adjusting the defensive floor based on whether the market is cooling off from a peak or bottoming out from a crash.

---

## 📈 Intelligence Modules

### 1. Dual-Window Anomaly Detector (`dual_window_detector_v5.py`)
A streaming-agnostic engine designed for real-time tick processing. It utilizes nested windows to distinguish between:
*   **High-Frequency Spikes**: Localized volatility anomalies (Z-score 3.0+).
*   **Low-Frequency Drifts**: Long-term structural mean shifts that signal regime transitions.

### 2. Regime Transition Study (`regime_transition_study.py`)
A deep-dive analysis into Bitcoin's structural DNA from 2018 to 2025.
*   **Transition Probability Matrix**: Maps the likelihood of jumping between regimes (e.g., UP → NORMAL vs. UP → DOWN).
*   **Persistence Analytics**: Calculates the mathematical expectancy of regime durations in days.
*   **State Autocorrelation (ACF)**: Quantifies the "memory" of market states across multiple time horizons.

### 3. Stress Test & Friction Audit (`btc_stress_test.py`)
A brutal, high-fidelity backtest that enforces institutional-grade constraints:
*   **10 bps Commission** + **2 bps Slippage** on every trade.
*   **Strict Causality**: Zero-lookahead execution. Actions for bar `t` are determined solely by data available at `t-1`.

---

## 🚀 Usage Guide

### Installation
Ensure your environment has the following dependencies:
```bash
pip install numpy matplotlib seaborn pandas
```

### 1. Data Sanitization
First, clean and validate the raw ticker data:
```bash
python prepare_btc_data.py
```
*Creates `btc_close_clean.npy` (binary) and `btc_gaps_report.csv`.*

### 2. Structural Analysis
Generate the regime transition matrices and persistence reports:
```bash
python regime_transition_study.py
```

### 3. Strategy Validation
Run the performance simulation and stress tests:
```bash
python btc_adaptive_exposure.py
python btc_stress_test.py
```

---

## 📊 Visual Assets
The framework generates several key visualizations:
*   `adaptive_exposure_performance.png`: Equity curves vs. Buy & Hold.
*   `regime_transition_study.png`: Heatmaps and persistence histograms.
*   `btc_regime_overlay.png`: Price action color-coded by detected regime.

---

## 📄 License
This project is for research purposes only. Not financial advice. Licensed under the MIT License.
