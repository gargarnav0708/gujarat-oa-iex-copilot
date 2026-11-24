"""
Command-Line Interface (CLI) for the Gujarat Open Access + IEX AI Bidding Co-Pilot.
"""

import argparse
import json
import os
from .config import ConsumerProfile, GERCConfig, OAMode
from .dashboard import generate_html_dashboard
from .data_loader import load_iex_dataset
from .forecaster import IEXPriceForecaster
from .optimizer import GujaratOABiddingSimulator
from .weather import GujaratWeatherClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gujarat Open Access (GERC) + IEX AI Market Intelligence & Bidding Simulator"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to IEX_Market_Data.csv (from IEXInsiderMCP).",
    )
    parser.add_argument(
        "--base-load-mw",
        type=float,
        default=4.0,
        help="Continuous unshiftable industrial base load in MW (default: 4.0 MW).",
    )
    parser.add_argument(
        "--flex-load-mw",
        type=float,
        default=1.0,
        help="Shiftable flexible load (HVAC/pumping/furnace) in MW (default: 1.0 MW).",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2025-06-01",
        help="Train/Backtest split date YYYY-MM-DD (default: 2025-06-01).",
    )
    parser.add_argument(
        "--demo-date",
        type=str,
        default="2025-07-20",
        help="Target date YYYY-MM-DD for 24-hour granular bidding trace (default: 2025-07-20).",
    )
    parser.add_argument(
        "--no-live-weather",
        action="store_true",
        help="Skip live Open-Meteo API call and use cached CSV or Gujarat climatological model.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run walk-forward TimeSeriesSplit (48h embargo gap) hyperparameter tuning with early stopping.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=10,
        help="Number of walk-forward hyperparameter tuning trials when --tune is enabled (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to write JSON results and interactive HTML dashboard (default: ./outputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(" GUJARAT OPEN ACCESS (GERC) + IEX AI BIDDING CO-PILOT (WITH WEATHER API) ")
    print("=" * 80)

    print("\n[1/5] Loading & synchronizing 15-minute IEX DAM, RTM, and GDAM records...")
    df_all = load_iex_dataset(args.data_path)
    print(
        f"      Loaded {len(df_all):,} synchronized 15-minute blocks "
        f"({df_all['Date'].min().date()} to {df_all['Date'].max().date()})."
    )

    print("\n[2/5] Integrating Gujarat Weather & Renewable Signals (Open-Meteo API: Kutch RE Hub)...")
    weather_client = GujaratWeatherClient()
    df_all, weather_meta = weather_client.enrich_iex_dataframe(
        df_all, use_live_api=not args.no_live_weather
    )
    print(f"      Weather Source  : {weather_meta['weather_source']}")
    print(f"      RE Hub Location : {weather_meta['re_hub_coordinates']}")
    print(
        f"      Gujarat Averages: Temp {weather_meta['avg_temp_c']}°C | "
        f"Solar GHI {weather_meta['avg_solar_wm2']} W/m² | "
        f"100m Wind {weather_meta['avg_wind_100m_kmh']} km/h"
    )

    print("\n[3/5] Training Weather-Aware Multi-Quantile XGBoost Forecasters (DAM P25/P50/P75 & RTM P50)...")
    if args.tune:
        print(f"      Running Walk-Forward TimeSeriesSplit (gap=192 blocks) Tuning ({args.tune_trials} trials)...")
    forecaster = IEXPriceForecaster()
    test_df, ml_metrics = forecaster.fit_and_evaluate(
        df_all, split_date=args.split_date, tune=args.tune, n_trials=args.tune_trials
    )
    print(
        f"      Backtest Window : {ml_metrics['test_start']} to {ml_metrics['test_end']} "
        f"({ml_metrics['test_blocks']:,} blocks | {ml_metrics['num_dam_features']} features)"
    )
    print(f"      Active Params   : {ml_metrics['best_hyperparameters']}")
    print(
        f"      DAM MAE         : ₹{ml_metrics['dam_mae_rs_kwh']:.3f}/kWh | "
        f"Pinball (P25/P50/P75): {ml_metrics['dam_pinball_p25']:.3f} / {ml_metrics['dam_pinball_p50']:.3f} / {ml_metrics['dam_pinball_p75']:.3f} | "
        f"P75 Clearance: {ml_metrics['p75_clearance_reliability_pct']:.1f}%"
    )
    print(f"      RTM MAE         : ₹{ml_metrics['rtm_mae_rs_kwh']:.3f}/kWh")

    print("\n[4/5] Simulating Gujarat (GERC) Open Access bidding & load shifting...")
    gerc = GERCConfig()
    consumer = ConsumerProfile(base_load_mw=args.base_load_mw, flex_load_mw=args.flex_load_mw)
    simulator = GujaratOABiddingSimulator(gerc_config=gerc, consumer=consumer)

    results = {
        "consumer_profile": {
            "name": consumer.name,
            "base_load_mw": consumer.base_load_mw,
            "flex_load_mw": consumer.flex_load_mw,
            "total_avg_load_mw": consumer.total_avg_load_mw,
        },
        "weather_metadata": weather_meta,
        "ml_metrics": ml_metrics,
        "scenarios": {},
    }

    for mode in [
        OAMode.EHV_66KV_STANDARD,
        OAMode.HT_11_33KV_STANDARD,
        OAMode.HT_11_33KV_GREEN_CAPTIVE,
    ]:
        sim_df, summary = simulator.simulate_mode(test_df, mode=mode)
        results["scenarios"][mode.value] = summary

        print(f"\n  --> Profile: {mode.value}")
        print(
            f"      DISCOM HTP-I Baseline    : ₹{summary['avg_discom_rate']:.2f}/kWh "
            f"(₹{summary['monthly_cost_discom_lakhs']:.2f} Lakhs/mo)"
        )
        print(
            f"      Naive Fixed OA Bidding   : ₹{summary['avg_naive_oa_rate']:.2f}/kWh "
            f"(₹{summary['monthly_cost_naive_lakhs']:.2f} Lakhs/mo | {summary['savings_pct_naive']:.2f}% saved)"
        )
        print(
            f"      AI Co-Pilot (No Shift)   : ₹{summary['avg_ai_oa_rate']:.2f}/kWh "
            f"(₹{summary['monthly_cost_ai_lakhs']:.2f} Lakhs/mo | {summary['savings_pct_ai']:.2f}% saved)"
        )
        print(
            f"      AI Co-Pilot + Load Shift : ₹{summary['avg_ai_shift_rate']:.2f}/kWh "
            f"(₹{summary['monthly_cost_ai_shift_lakhs']:.2f} Lakhs/mo | {summary['savings_pct_ai_shift']:.2f}% saved)"
        )
        print(
            f"      Net Savings vs. DISCOM   : ₹{summary['monthly_savings_ai_shift_lakhs']:.2f} Lakhs/mo "
            f"(₹{summary['annual_savings_ai_shift_crores']:.2f} Cr/yr)"
        )
        print(
            f"      Optimal Procurement Mix  : {summary['dam_energy_share_pct']}% DAM + "
            f"{summary['rtm_energy_share_pct']}% RTM + {summary['discom_energy_share_pct']}% DISCOM"
        )

        if mode == OAMode.EHV_66KV_STANDARD:
            results["demo_day"] = simulator.extract_hourly_demo_day(sim_df, target_date=args.demo_date)

    json_path = os.path.join(args.output_dir, "gujarat_simulation_results.json")
    html_path = os.path.join(args.output_dir, "gujarat_oa_dashboard.html")

    print("\n[5/5] Exporting JSON report and interactive HTML dashboard...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    generate_html_dashboard(results, html_path)

    print(f"      JSON Results   : {os.path.abspath(json_path)}")
    print(f"      HTML Dashboard : {os.path.abspath(html_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
