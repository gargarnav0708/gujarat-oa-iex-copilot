"""
Unit Tests for the Gujarat Open Access (GERC) Tariff, Breakeven, Weather API Client, and Optimizer Engine.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gujarat_oa_copilot.config import ConsumerProfile, GERCConfig, OAMode
from gujarat_oa_copilot.optimizer import GujaratOABiddingSimulator
from gujarat_oa_copilot.weather import GujaratWeatherClient


class TestGERCConfigAndOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.gerc = GERCConfig()

    def test_discom_tou_tariffs(self) -> None:
        # Normal / Night hours: 4.45 + 2.85 = 7.30
        self.assertAlmostEqual(self.gerc.get_discom_variable_tariff(2.0), 7.30, places=2)
        # Morning Peak (07:00-11:00) & Evening Peak (18:00-22:00): 7.30 + 0.85 = 8.15
        self.assertAlmostEqual(self.gerc.get_discom_variable_tariff(8.5), 8.15, places=2)
        self.assertAlmostEqual(self.gerc.get_discom_variable_tariff(19.0), 8.15, places=2)
        # Midday Solar Window (11:00-17:00): 7.30 - 0.45 = 6.85
        self.assertAlmostEqual(self.gerc.get_discom_variable_tariff(13.0), 6.85, places=2)

    def test_breakeven_identity_66kv(self) -> None:
        # At the breakeven MCP, OA landed cost must exactly equal the DISCOM tariff
        for hour in [2.0, 9.0, 14.0, 20.0]:
            discom_rate = self.gerc.get_discom_variable_tariff(hour)
            be_mcp = self.gerc.get_breakeven_mcp(discom_rate, mode=OAMode.EHV_66KV_STANDARD)
            landed = self.gerc.get_oa_landed_cost(be_mcp, mode=OAMode.EHV_66KV_STANDARD)
            self.assertAlmostEqual(landed, discom_rate, places=5)

    def test_breakeven_identity_11_33kv(self) -> None:
        discom_rate = self.gerc.get_discom_variable_tariff(9.0)
        be_mcp = self.gerc.get_breakeven_mcp(discom_rate, mode=OAMode.HT_11_33KV_STANDARD)
        landed = self.gerc.get_oa_landed_cost(be_mcp, mode=OAMode.HT_11_33KV_STANDARD)
        self.assertAlmostEqual(landed, discom_rate, places=5)

    def test_weather_client_enrichment(self) -> None:
        dates = pd.date_range("2025-07-01", periods=1, freq="D")
        rows = [{"Date": dates[0], "Block_Index": b, "Hour_Float": (b - 1) * 0.25, "DAM_MCP": 3.0, "RTM_MCP": 3.0} for b in range(1, 97)]
        sample_df = pd.DataFrame(rows)
        client = GujaratWeatherClient()
        enriched, meta = client.enrich_iex_dataframe(sample_df, use_live_api=False)
        for col in client.WEATHER_FEATURES:
            self.assertIn(col, enriched.columns)
            self.assertFalse(enriched[col].isna().any())
        # Daytime solar radiation should exceed midnight solar radiation
        midday_solar = enriched.loc[enriched["Hour_Float"] == 12.5, "Solar_Radiation_Wm2"].iloc[0]
        midnight_solar = enriched.loc[enriched["Hour_Float"] == 1.0, "Solar_Radiation_Wm2"].iloc[0]
        self.assertGreater(midday_solar, midnight_solar)

    def test_ai_strategy_never_exceeds_discom_cost(self) -> None:
        # Build a synthetic 2-day (192-block) test dataframe
        dates = pd.date_range("2025-07-01", periods=2, freq="D")
        rows = []
        for d in dates:
            for b in range(1, 97):
                hr = (b - 1) * 0.25
                dam_mcp = 2.10 if 8.0 <= hr <= 16.0 else 5.50
                rtm_mcp = 1.95 if 12.0 <= hr <= 15.0 else 5.80
                rows.append(
                    {
                        "Date": d,
                        "Block_Index": b,
                        "Hour_Float": hr,
                        "DAM_MCP": dam_mcp,
                        "RTM_MCP": rtm_mcp,
                        "Pred_DAM_P25": dam_mcp - 0.3,
                        "Pred_DAM_P50": dam_mcp,
                        "Pred_DAM_P75": dam_mcp + 0.3,
                        "Pred_RTM_P50": rtm_mcp,
                    }
                )
        test_df = pd.DataFrame(rows)
        sim = GujaratOABiddingSimulator(self.gerc, ConsumerProfile(base_load_mw=4.0, flex_load_mw=1.0))
        sim_df, summary = sim.simulate_mode(test_df, mode=OAMode.EHV_66KV_STANDARD)

        # AI Co-Pilot effective rate must be <= DISCOM rate in every single 15-min block
        self.assertTrue(np.all(sim_df["Rate_S2_AI"] <= sim_df["DISCOM_Rate"] + 1e-6))
        self.assertGreater(summary["savings_pct_ai_shift"], summary["savings_pct_ai"])
        self.assertGreater(summary["savings_pct_ai"], 0.0)


if __name__ == "__main__":
    unittest.main()
