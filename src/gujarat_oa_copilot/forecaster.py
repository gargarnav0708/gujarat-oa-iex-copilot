"""
AI Forecasting Layer: Multi-Quantile XGBoost Models for IEX DAM and RTM Prices.
Integrates historical IEX market lags, cyclic seasonality, and Gujarat weather/RE
signals (Solar Irradiance, 100m Wind Speed, Cloud Cover, Cooling Degree Load Index).
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class IEXPriceForecaster:
    """
    Trains and evaluates Day-Ahead Market (DAM) and Real-Time Market (RTM)
    quantile regression models without lookahead leakage.
    """

    BASE_DAM_FEATURES = [
        "Block_Index",
        "Hour_Float",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "Sin_Hour",
        "Cos_Hour",
        "DAM_MCP_Lag2d",
        "DAM_MCP_Lag3d",
        "DAM_MCP_Lag7d",
        "RTM_MCP_Lag2d",
        "RTM_MCP_Lag7d",
        "DAM_Demand_Lag2d",
        "DAM_Supply_Lag2d",
        "Demand_Supply_Ratio_Lag2d",
        "Demand_Supply_Ratio_Lag7d",
        "DAM_MCP_Roll7d_Block",
        "DAM_MCP_Roll7d_Std",
    ]

    WEATHER_FEATURES = [
        "Temp_C",
        "Humidity_Pct",
        "Solar_Radiation_Wm2",
        "Wind_Speed_100m_kmh",
        "Cloud_Cover_Pct",
        "Cooling_Degree_Load_Index",
        "Gujarat_Solar_Gen_Index",
        "Gujarat_Wind_Gen_Index",
        "Gujarat_Total_RE_Index",
    ]

    def __init__(self, n_estimators: int = 260, max_depth: int = 6, learning_rate: float = 0.05):
        self.model_dam_p25 = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:quantileerror",
            quantile_alpha=0.25,
            random_state=42,
            n_jobs=-1,
        )
        self.model_dam_p50 = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:absoluteerror",
            random_state=42,
            n_jobs=-1,
        )
        self.model_dam_p75 = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:quantileerror",
            quantile_alpha=0.75,
            random_state=42,
            n_jobs=-1,
        )
        self.model_rtm_p50 = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:absoluteerror",
            random_state=42,
            n_jobs=-1,
        )
        self.active_dam_features: List[str] = []
        self.active_rtm_features: List[str] = []

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Creates temporal, cyclic, lag, supply-demand ratio, and weather interaction features."""
        data = df.copy()
        data["DayOfWeek"] = data["Date"].dt.dayofweek
        data["Month"] = data["Date"].dt.month
        data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)
        data["Sin_Hour"] = np.sin(2 * np.pi * data["Hour_Float"] / 24.0)
        data["Cos_Hour"] = np.cos(2 * np.pi * data["Hour_Float"] / 24.0)

        for lag_days in [2, 3, 7]:
            shift_blocks = lag_days * 96
            data[f"DAM_MCP_Lag{lag_days}d"] = data["DAM_MCP"].shift(shift_blocks)
            data[f"RTM_MCP_Lag{lag_days}d"] = data["RTM_MCP"].shift(shift_blocks)
            data[f"DAM_Demand_Lag{lag_days}d"] = data["DAM_IEX_Demand"].shift(shift_blocks)
            data[f"DAM_Supply_Lag{lag_days}d"] = data["DAM_IEX_Supply"].shift(shift_blocks)
            data[f"Demand_Supply_Ratio_Lag{lag_days}d"] = data[f"DAM_Demand_Lag{lag_days}d"] / (
                data[f"DAM_Supply_Lag{lag_days}d"] + 0.1
            )

        data["DAM_MCP_Roll7d_Block"] = data.groupby("Block_Index")["DAM_MCP"].transform(
            lambda x: x.shift(2).rolling(7, min_periods=3).mean()
        )
        data["DAM_MCP_Roll7d_Std"] = data.groupby("Block_Index")["DAM_MCP"].transform(
            lambda x: x.shift(2).rolling(7, min_periods=3).std()
        )
        data["RTM_MCP_Roll7d_Block"] = data.groupby("Block_Index")["RTM_MCP"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=3).mean()
        )
        return data

    def fit_and_evaluate(
        self, df: pd.DataFrame, split_date: str = "2025-06-01", extra_features: List[str] | None = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineers features (automatically including Gujarat weather features if present in df
        plus any extra_features like Prophet/LSTM embeddings), trains all 4 XGBoost models,
        and returns the backtest DataFrame with metrics.
        """
        feat_df = self.engineer_features(df)

        available_weather = [c for c in self.WEATHER_FEATURES if c in feat_df.columns]
        extra = [c for c in (extra_features or []) if c in feat_df.columns]

        self.active_dam_features = self.BASE_DAM_FEATURES + available_weather + extra
        self.active_rtm_features = self.active_dam_features + [
            "DAM_MCP",
            "DAM_IEX_Demand",
            "DAM_IEX_Supply",
            "RTM_MCP_Roll7d_Block",
        ]

        clean = feat_df.dropna(subset=self.active_rtm_features).reset_index(drop=True)

        cutoff = pd.Timestamp(split_date)
        train_df = clean[clean["Date"] < cutoff].copy()
        test_df = clean[clean["Date"] >= cutoff].copy()

        self.model_dam_p25.fit(train_df[self.active_dam_features], train_df["DAM_MCP"])
        self.model_dam_p50.fit(train_df[self.active_dam_features], train_df["DAM_MCP"])
        self.model_dam_p75.fit(train_df[self.active_dam_features], train_df["DAM_MCP"])
        self.model_rtm_p50.fit(train_df[self.active_rtm_features], train_df["RTM_MCP"])

        test_df["Pred_DAM_P25"] = np.clip(
            self.model_dam_p25.predict(test_df[self.active_dam_features]), 0.1, 12.0
        )
        test_df["Pred_DAM_P50"] = np.clip(
            self.model_dam_p50.predict(test_df[self.active_dam_features]), 0.1, 12.0
        )
        test_df["Pred_DAM_P75"] = np.clip(
            self.model_dam_p75.predict(test_df[self.active_dam_features]), 0.1, 12.0
        )
        test_df["Pred_RTM_P50"] = np.clip(
            self.model_rtm_p50.predict(test_df[self.active_rtm_features]), 0.1, 12.0
        )

        dam_mae = mean_absolute_error(test_df["DAM_MCP"], test_df["Pred_DAM_P50"])
        dam_rmse = root_mean_squared_error(test_df["DAM_MCP"], test_df["Pred_DAM_P50"])
        rtm_mae = mean_absolute_error(test_df["RTM_MCP"], test_df["Pred_RTM_P50"])
        p75_clearance = np.mean(test_df["DAM_MCP"] <= (test_df["Pred_DAM_P75"] + 0.15)) * 100.0

        # Top feature importances from DAM P50 model
        importances = dict(
            sorted(
                zip(
                    self.active_dam_features,
                    [round(float(x), 4) for x in self.model_dam_p50.feature_importances_],
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        )

        metrics = {
            "train_blocks": int(len(train_df)),
            "test_blocks": int(len(test_df)),
            "train_start": str(train_df["Date"].min().date()),
            "train_end": str(train_df["Date"].max().date()),
            "test_start": str(test_df["Date"].min().date()),
            "test_end": str(test_df["Date"].max().date()),
            "weather_features_included": len(available_weather) > 0,
            "num_dam_features": len(self.active_dam_features),
            "dam_mae_rs_kwh": round(float(dam_mae), 3),
            "dam_rmse_rs_kwh": round(float(dam_rmse), 3),
            "rtm_mae_rs_kwh": round(float(rtm_mae), 3),
            "p75_clearance_reliability_pct": round(float(p75_clearance), 2),
            "top_10_dam_feature_importances": importances,
        }
        return test_df, metrics
