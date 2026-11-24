"""
Gujarat Open Access Bidding & Flexible Load Shifting Optimizer.
Simulates 4 distinct procurement strategies across 15-minute blocks:
  - Strategy 0: 100% DISCOM Default (HTP-I Tariff + FPPPA + ToU Peak/Solar)
  - Strategy 1: Naive Rule-of-Thumb OA Bidding (Static cap, 09:00-17:00)
  - Strategy 2: AI Co-Pilot Laddered Bidding + DAM/RTM Arbitrage
  - Strategy 3: AI Co-Pilot + Automated Flexible Load Shifting (Evening Peak -> Midday Solar Dip)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from .config import ConsumerProfile, GERCConfig, OAMode


class GujaratOABiddingSimulator:
    """Simulates and backtests Open Access bidding and load shifting in Gujarat."""

    def __init__(self, gerc_config: GERCConfig | None = None, consumer: ConsumerProfile | None = None):
        self.gerc = gerc_config or GERCConfig()
        self.consumer = consumer or ConsumerProfile()

    def simulate_mode(
        self, test_df: pd.DataFrame, mode: OAMode | str | None = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Runs the 4-strategy comparison for a specific Gujarat OA connection mode."""
        active_mode = mode or self.consumer.oa_mode
        mode_str = active_mode.value if isinstance(active_mode, OAMode) else str(active_mode)

        df = test_df.copy()
        block_hours = 0.25  # 15-minute block

        # 1. Regulatory Tariff & Breakeven Calculations
        df["DISCOM_Rate"] = df["Hour_Float"].apply(self.gerc.get_discom_variable_tariff)
        df["Breakeven_MCP"] = df["DISCOM_Rate"].apply(
            lambda r: self.gerc.get_breakeven_mcp(r, mode=active_mode)
        )
        df["DAM_Landed_Cost"] = df["DAM_MCP"].apply(
            lambda m: self.gerc.get_oa_landed_cost(m, mode=active_mode)
        )
        df["RTM_Landed_Cost"] = df["RTM_MCP"].apply(
            lambda m: self.gerc.get_oa_landed_cost(m, mode=active_mode)
        )

        df["Standard_Load_kWh"] = self.consumer.total_avg_load_mw * 1000.0 * block_hours

        # --- Strategy 0: 100% DISCOM Baseline ---
        df["Cost_S0_DISCOM"] = df["Standard_Load_kWh"] * df["DISCOM_Rate"]

        # --- Strategy 1: Naive Human / Trader Fixed-Cap Bidding ---
        naive_bid_cap = 3.60
        in_day_window = (df["Hour_Float"] >= 9.0) & (df["Hour_Float"] < 17.0)
        naive_clears_dam = in_day_window & (df["DAM_MCP"] <= naive_bid_cap)
        naive_rtm_try = in_day_window & (~naive_clears_dam) & (df["RTM_MCP"] <= 4.20)
        df["Rate_S1_Naive"] = np.where(
            naive_clears_dam,
            df["DAM_Landed_Cost"],
            np.where(naive_rtm_try, df["RTM_Landed_Cost"], df["DISCOM_Rate"]),
        )
        df["Cost_S1_Naive"] = df["Standard_Load_kWh"] * df["Rate_S1_Naive"]

        # --- Strategy 2: AI Co-Pilot Laddered Bidding + DAM/RTM Arbitrage ---
        bid_ceiling = np.maximum(0.0, df["Breakeven_MCP"] - 0.02)
        rtm_arbitrage_signal = (df["Pred_RTM_P50"] < (df["Pred_DAM_P50"] - 0.25)) & (
            df["Pred_RTM_P50"] < bid_ceiling
        )
        dam_share = np.where(rtm_arbitrage_signal, 0.50, 1.00)

        dam_clears = df["DAM_MCP"] <= bid_ceiling
        rtm_clears = df["RTM_MCP"] <= bid_ceiling

        cleared_dam_share = np.where(dam_clears, dam_share, 0.0)
        remaining_for_rtm = 1.0 - cleared_dam_share
        cleared_rtm_share = np.where(rtm_clears, remaining_for_rtm, 0.0)
        discom_fallback_share = 1.0 - cleared_dam_share - cleared_rtm_share

        df["Rate_S2_AI"] = (
            cleared_dam_share * df["DAM_Landed_Cost"]
            + cleared_rtm_share * df["RTM_Landed_Cost"]
            + discom_fallback_share * df["DISCOM_Rate"]
        )
        df["Cost_S2_AI"] = df["Standard_Load_kWh"] * df["Rate_S2_AI"]
        df["OA_Share_S2"] = cleared_dam_share + cleared_rtm_share
        df["DAM_Share_S2"] = cleared_dam_share
        df["RTM_Share_S2"] = cleared_rtm_share

        # --- Strategy 3: AI Co-Pilot + Flexible Load Shifting ---
        df["Shifted_Load_kWh"] = df["Standard_Load_kWh"].copy()
        shift_kwh_per_block = self.consumer.flex_load_mw * 1000.0 * block_hours

        for _, day_group in df.groupby("Date"):
            if len(day_group) < 96:
                continue
            evening_peak_idx = day_group[
                (day_group["Hour_Float"] >= 18.0) & (day_group["Hour_Float"] < 22.0)
            ].index
            candidate_idx = (
                day_group[(day_group["Hour_Float"] >= 10.0) & (day_group["Hour_Float"] < 17.0)]
                .sort_values("Pred_DAM_P50")
                .index[: len(evening_peak_idx)]
            )
            if len(evening_peak_idx) == len(candidate_idx):
                df.loc[evening_peak_idx, "Shifted_Load_kWh"] -= shift_kwh_per_block
                df.loc[candidate_idx, "Shifted_Load_kWh"] += shift_kwh_per_block

        df["Cost_S3_AI_Shift"] = df["Shifted_Load_kWh"] * df["Rate_S2_AI"]

        # Summary Aggregation
        total_kwh = df["Standard_Load_kWh"].sum()
        num_days = max(1, df["Date"].nunique())
        monthly_scale = 30.0 / num_days
        annual_scale = 365.0 / num_days

        c0 = df["Cost_S0_DISCOM"].sum()
        c1 = df["Cost_S1_Naive"].sum()
        c2 = df["Cost_S2_AI"].sum()
        c3 = df["Cost_S3_AI_Shift"].sum()

        summary = {
            "mode": mode_str,
            "regulatory_stack": self.gerc.get_cost_stack_breakdown(active_mode),
            "num_days": int(num_days),
            "total_mu": round(float(total_kwh / 1e6), 3),
            "avg_discom_rate": round(float(c0 / total_kwh), 2),
            "avg_naive_oa_rate": round(float(c1 / total_kwh), 2),
            "avg_ai_oa_rate": round(float(c2 / total_kwh), 2),
            "avg_ai_shift_rate": round(float(c3 / total_kwh), 2),
            "oa_energy_share_pct": round(float(df["OA_Share_S2"].mean() * 100.0), 1),
            "dam_energy_share_pct": round(float(cleared_dam_share.mean() * 100.0), 1),
            "rtm_energy_share_pct": round(float(cleared_rtm_share.mean() * 100.0), 1),
            "discom_energy_share_pct": round(float(discom_fallback_share.mean() * 100.0), 1),
            "monthly_cost_discom_lakhs": round(float(c0 * monthly_scale / 1e5), 2),
            "monthly_cost_naive_lakhs": round(float(c1 * monthly_scale / 1e5), 2),
            "monthly_cost_ai_lakhs": round(float(c2 * monthly_scale / 1e5), 2),
            "monthly_cost_ai_shift_lakhs": round(float(c3 * monthly_scale / 1e5), 2),
            "monthly_savings_naive_lakhs": round(float((c0 - c1) * monthly_scale / 1e5), 2),
            "monthly_savings_ai_lakhs": round(float((c0 - c2) * monthly_scale / 1e5), 2),
            "monthly_savings_ai_shift_lakhs": round(float((c0 - c3) * monthly_scale / 1e5), 2),
            "ai_alpha_over_naive_lakhs_mo": round(float((c1 - c3) * monthly_scale / 1e5), 2),
            "annual_savings_ai_shift_crores": round(float((c0 - c3) * annual_scale / 1e7), 2),
            "savings_pct_naive": round(float((c0 - c1) / c0 * 100.0), 2),
            "savings_pct_ai": round(float((c0 - c2) / c0 * 100.0), 2),
            "savings_pct_ai_shift": round(float((c0 - c3) / c0 * 100.0), 2),
        }
        return df, summary

    @staticmethod
    def extract_hourly_demo_day(sim_df: pd.DataFrame, target_date: str = "2025-07-20") -> Dict[str, Any]:
        """Extracts a 24-hour aggregated schedule for a single demonstration day."""
        day_df = sim_df[sim_df["Date"] == pd.Timestamp(target_date)].copy()
        if day_df.empty:
            target_date = str(sim_df["Date"].unique()[min(30, len(sim_df["Date"].unique()) - 1)].date())
            day_df = sim_df[sim_df["Date"] == pd.Timestamp(target_date)].copy()

        day_df["Hour"] = day_df["Hour_Float"].astype(int)
        hourly = (
            day_df.groupby("Hour")
            .agg(
                DAM_MCP=("DAM_MCP", "mean"),
                RTM_MCP=("RTM_MCP", "mean"),
                Pred_DAM_P50=("Pred_DAM_P50", "mean"),
                Pred_DAM_P75=("Pred_DAM_P75", "mean"),
                Pred_RTM_P50=("Pred_RTM_P50", "mean"),
                DISCOM_Rate=("DISCOM_Rate", "mean"),
                Breakeven_MCP=("Breakeven_MCP", "mean"),
                DAM_Landed_Cost=("DAM_Landed_Cost", "mean"),
                RTM_Landed_Cost=("RTM_Landed_Cost", "mean"),
                Effective_AI_Rate=("Rate_S2_AI", "mean"),
                OA_Share=("OA_Share_S2", "mean"),
                DAM_Share=("DAM_Share_S2", "mean"),
                RTM_Share=("RTM_Share_S2", "mean"),
                Shifted_Load_MW=("Shifted_Load_kWh", lambda x: x.sum() / 1000.0),
            )
            .reset_index()
        )
        return {
            "date": target_date,
            "hourly": hourly.round(2).to_dict(orient="records"),
        }
