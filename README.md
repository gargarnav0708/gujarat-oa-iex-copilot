#  Gujarat Open Access (GERC) + IEX AI Bidding Co-Pilot

An AI-powered electricity market intelligence, price forecasting, and Open Access (OA) bidding & load-shifting simulator tailored for Commercial & Industrial (C&I) consumers in **Gujarat** (`DGVCL`, `PGVCL`, `UGVCL`, `MGVCL`, and `GETCO` EHV).

Built on top of the **[IEXInsiderMCP](https://github.com/divyeshBhartiya/IEXInsiderMCP)** dataset (**289,214 15-minute blocks** across **DAM**, **RTM**, and **GDAM** from January 2023 to September 2025) and live/historical **Open-Meteo Weather API** telemetry for Gujarat's **Khavda / Kutch Ultra-Mega Renewable Energy Park (`23.83°N, 69.73°E`)**, this repository implements:

1. **Live & Historical Weather API Integration (`GujaratWeatherClient`):**
   - Queries the **Open-Meteo Historical Archive API** and **7-Day Forecast API** for Gujarat (`23.83°N, 69.73°E` Kutch RE Hub & `23.02°N, 72.57°E` Ahmedabad GIDC Industrial Hub) with automatic CSV caching (`data/gujarat_weather_cache.csv`) and an offline climatological physics fallback.
   - Extracts **Solar Irradiance (`W/m²`)**, **100m Wind Turbine Hub-Height Speed (`km/h`)**, **Cloud Cover (`%`)**, **Ambient Temperature (`°C`)**, **Relative Humidity (`%`)**, **Cooling Degree Load Index**, and normalized **Gujarat Solar & Wind Generation Indices**.
2. **Weather-Aware Multi-Quantile XGBoost Price Forecasting (`IEXPriceForecaster`):**
   - Predicts Day-Ahead Market (`DAM P25`, `P50`, `P75`) and intraday Real-Time Market (`RTM P50`) clearing prices across 27+ market and meteorological features with zero lookahead leakage.
3. **Hybrid Prophet + PyTorch Sequence LSTM + XGBoost Quantile Research Lab (`notebooks/01_Prophet_LSTM_XGBoost_Weather_Experiments.ipynb`):**
   - Full experimental Jupyter Notebook benchmarking **Prophet additive Fourier/Changepoint decomposition**, a **2-layer PyTorch LSTM (`IEXSequenceLSTM`)** for long-term sequential dependencies, and a **Hybrid Stacked Meta-Ensemble (Prophet + PyTorch LSTM + Weather-Aware Multi-Quantile XGBoost)**.
4. **Official Gujarat (GERC) Open Access Cost Stack Engine (`GERCConfig`):**
   - Accounts for HTP-I DISCOM Base Energy Charges, FPPPA, Time-of-Use (ToU) Peak surcharges (`+₹0.85/kWh`) & Solar concessions (`-₹0.45/kWh`), Cross-Subsidy Surcharge (`CSS ₹1.79/kWh`), Additional Surcharge (`AS ₹1.36/kWh`), GETCO STU transmission charges (`₹0.39/kWh`), DISCOM wheeling (`₹0.60/kWh`), and cumulative technical losses (`ISTS 3.20%`, `STU 3.35%`, `Wheeling 7.50%`).
5. **Dynamic 15-Minute Breakeven Bid Laddering, DAM/RTM Arbitrage & Automated 1 MW Load Shifting (`GujaratOABiddingSimulator`):**
   - Guarantees zero overpayment traps relative to the consumer's ToU DISCOM tariff and shifts flexible industrial load out of Evening Peak hours (`18:00–22:00`) into the lowest-landed-cost midday Solar/IEX dip blocks (`10:00–16:00`).
6. **Interactive Self-Contained HTML Dashboard:**
   - Exports an interactive Chart.js dashboard (`outputs/gujarat_oa_dashboard.html`) and structured JSON results (`outputs/gujarat_simulation_results.json`).

---

## Headline Backtest Results (June–September 2025, 11,616 15-Min Blocks)

For a **5 MW Industrial Plant in Gujarat** (`4 MW` fixed base load + `1 MW` flexible shiftable load, consuming `3.60 MU/month`) using the **Weather-Aware Multi-Quantile XGBoost Co-Pilot**:

| Procurement Strategy | 66 kV EHV Standard OA<br/>*(Pays Full CSS + AS)* | 11/33 kV HT Standard OA<br/>*(Pays CSS + AS + Wheeling)* | 11/33 kV Green / Captive OA<br/>*(CSS & AS Waived)* |
| :--- | :---: | :---: | :---: |
| **0. 100% DISCOM Baseline (HTP-I)** | `₹7.47 / kWh`<br/>*(₹268.95 Lakhs / mo)* | `₹7.47 / kWh`<br/>*(₹268.95 Lakhs / mo)* | `₹7.47 / kWh`<br/>*(₹268.95 Lakhs / mo)* |
| **1. Naive Fixed OA Bidding**<br/>*(Static ₹3.60 cap, 09:00–17:00)* | `₹6.93 / kWh`<br/>*(₹249.63 Lakhs / mo — 7.18% saved)* | `₹7.19 / kWh`<br/>*(₹258.71 Lakhs / mo — 3.81% saved)* | `₹6.04 / kWh`<br/>*(₹217.31 Lakhs / mo — 19.20% saved)* |
| **2. AI Co-Pilot (DAM + RTM Arbitrage)**<br/>*(Dynamic ToU breakeven + Weather RTM split)* | `₹6.60 / kWh`<br/>*(₹237.67 Lakhs / mo — **11.63% saved**)* | `₹6.98 / kWh`<br/>*(₹251.44 Lakhs / mo — **6.51% saved**)* | `₹4.50 / kWh`<br/>*(₹161.90 Lakhs / mo — **39.80% saved**)* |
| **3. AI Co-Pilot + 1 MW Load Shift**<br/>*(Shifts 1 MW out of Evening Peak)* | **`₹6.52 / kWh`**<br/>*(₹234.68 Lakhs / mo — **12.74% saved**)* | **`₹6.92 / kWh`**<br/>*(₹249.03 Lakhs / mo — **7.41% saved**)* | **`₹4.36 / kWh`**<br/>*(₹157.13 Lakhs / mo — **41.58% saved**)* |
| **Net Monthly Savings (vs. DISCOM)** | **`₹34.27 Lakhs / month`** | **`₹19.92 Lakhs / month`** | **`₹111.82 Lakhs / month`** |
| **Net Annual Savings (vs. DISCOM)** | **`₹4.17 Crores / year`** | **`₹2.42 Crores / year`** | **`₹13.60 Crores / year`** |
| **Optimal Energy Mix (DAM / RTM / DISCOM)** | `52.5% DAM` • `17.9% RTM`<br/>`29.6% DISCOM` | `33.4% DAM` • `11.6% RTM`<br/>`55.0% DISCOM` | `72.9% DAM` • `17.4% RTM`<br/>`9.7% DISCOM` |

---

## Repository Structure

```text
gujarat-oa-iex-copilot/
├── README.md                              # Project documentation, math, and CLI usage
├── requirements.txt                       # Python dependencies (XGBoost, PyTorch, Requests, Jupyter)
├── pyproject.toml                         # Standard PEP 621 package configuration
├── run_demo.py                            # Quick-start CLI runner script
├── notebooks/
│   └── 01_Prophet_LSTM_XGBoost_Weather_Experiments.ipynb  # Executed Prophet + PyTorch LSTM + Weather XGBoost Lab
├── src/
│   └── gujarat_oa_copilot/
│       ├── __init__.py                    # Package exports
│       ├── config.py                      # GERC HTP-I Tariff, ToU, CSS, AS, Losses & Breakeven engine
│       ├── data_loader.py                 # Synchronizes 289K 15-min DAM, RTM & GDAM records
│       ├── weather.py                     # Open-Meteo Archive & 7-Day Forecast API client for Gujarat RE Hub
│       ├── forecaster.py                  # Weather-aware multi-quantile XGBoost forecaster (P25/P50/P75)
│       ├── optimizer.py                   # 4-strategy Gujarat OA bidding & load-shift simulator
│       ├── dashboard.py                   # Interactive Chart.js HTML report generator
│       └── cli.py                         # Command-line interface implementation
├── tests/
│   └── test_copilot.py                    # Unit tests for GERC math, Weather API client & optimizer
├── data/
│   └── gujarat_weather_cache.csv          # Cached Open-Meteo hourly weather for Kutch RE Hub (2023-2025)
├── outputs/
│   ├── gujarat_oa_dashboard.html          # Generated interactive Chart.js dashboard
│   └── gujarat_simulation_results.json    # Complete backtest metrics & 24-hour trace JSON
└── IEXInsiderMCP/                         # Upstream IEXInsiderMCP repo containing IEX_Market_Data.csv
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or install as an editable Python package:
pip install -e .
```

### 2. Run the Full Backtest (with Live Open-Meteo Weather API) & Generate Dashboard

```bash
python3 run_demo.py
```

Or run with **Walk-Forward `TimeSeriesSplit` (`gap=192` blocks / 48-hour embargo) + Early Stopping Hyperparameter Optimization (`--tune`)**:

```bash
python3 run_demo.py \
  --tune \
  --tune-trials 10 \
  --base-load-mw 4.0 \
  --flex-load-mw 1.0 \
  --split-date 2025-06-01 \
  --demo-date 2025-07-20 \
  --output-dir outputs
```

### 3. Explore the Prophet + PyTorch LSTM + Weather-XGBoost Notebook

```bash
jupyter notebook notebooks/01_Prophet_LSTM_XGBoost_Weather_Experiments.ipynb
```

### 4. Run Unit Tests

```bash
python3 -m unittest discover -s tests -v
```
