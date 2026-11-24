"""
AI Forecasting Layer: Multi-Quantile XGBoost Models for IEX DAM and RTM Prices.
Integrates historical IEX market lags, cyclic seasonality, and Gujarat weather/RE
signals (Solar Irradiance, 100m Wind Speed, Cloud Cover, Cooling Degree Load Index).
Includes TimeSeriesSplit (Walk-Forward with 48-hour Embargo Gap) + Early Stopping
Hyperparameter Optimization minimizing Pinball (Quantile) Loss.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class IEXPriceForecaster:
    """
    Trains, tunes, and evaluates Day-Ahead Market (DAM) and Real-Time Market (RTM)
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

    DEFAULT_PARAMS = {
        "n_estimators": 225,
        "max_depth": 6,
        "learning_rate": 0.049,
        "min_child_weight": 10,
        "subsample": 0.72,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1675,
        "reg_lambda": 9.9729,
    }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = {**self.DEFAULT_PARAMS, **(params or {})}
        self.best_params: Dict[str, Any] = cfg
        self.tuning_history: List[Dict[str, Any]] = []
        self._init_models(cfg)
        self.active_dam_features: List[str] = []
        self.active_rtm_features: List[str] = []

    def _init_models(self, cfg: Dict[str, Any]) -> None:
        common = {
            "n_estimators": int(cfg.get("n_estimators", 280)),
            "max_depth": int(cfg.get("max_depth", 6)),
            "learning_rate": float(cfg.get("learning_rate", 0.045)),
            "min_child_weight": int(cfg.get("min_child_weight", 25)),
            "subsample": float(cfg.get("subsample", 0.85)),
            "colsample_bytree": float(cfg.get("colsample_bytree", 0.85)),
            "reg_alpha": float(cfg.get("reg_alpha", 0.1)),
            "reg_lambda": float(cfg.get("reg_lambda", 1.5)),
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model_dam_p25 = xgb.XGBRegressor(
            **common,
            objective="reg:quantileerror",
            quantile_alpha=0.25,
        )
        self.model_dam_p50 = xgb.XGBRegressor(
            **common,
            objective="reg:absoluteerror",
        )
        self.model_dam_p75 = xgb.XGBRegressor(
            **common,
            objective="reg:quantileerror",
            quantile_alpha=0.75,
        )
        self.model_rtm_p50 = xgb.XGBRegressor(
            **common,
            objective="reg:absoluteerror",
        )

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

    def tune_hyperparameters(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "DAM_MCP",
        alpha: float = 0.50,
        n_trials: int = 10,
        n_splits: int = 3,
        embargo_blocks: int = 192,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Finds optimal XGBoost hyperparameters using Walk-Forward TimeSeriesSplit
        with a 48-hour (192-block) embargo gap and early stopping, minimizing Pinball Loss.
        """
        X = train_df[feature_cols].values
        y = train_df[target_col].values

        # Ensure validation fold size fits small synthetic test DataFrames as well as full 84K-row datasets
        val_size = min(96 * 45, max(10, len(X) // (n_splits + 2)))
        gap_size = min(embargo_blocks, max(0, (len(X) // (n_splits + 2)) - val_size))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_size, test_size=val_size)
        rng = np.random.default_rng(42)

        best_score = float("inf")
        best_params = dict(self.DEFAULT_PARAMS)
        self.tuning_history = []

        # Candidate search grid (Trial 1 tests sensible baseline, subsequent trials explore space)
        for trial in range(1, n_trials + 1):
            if trial == 1:
                candidate = {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.045,
                    "min_child_weight": 25,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.5,
                }
            else:
                candidate = {
                    "n_estimators": 600,
                    "max_depth": int(rng.integers(4, 9)),
                    "learning_rate": float(np.round(rng.uniform(0.02, 0.08), 3)),
                    "min_child_weight": int(rng.choice([10, 20, 35, 50, 80])),
                    "subsample": float(np.round(rng.uniform(0.65, 0.95), 2)),
                    "colsample_bytree": float(np.round(rng.uniform(0.65, 0.95), 2)),
                    "reg_alpha": float(np.round(10 ** rng.uniform(-2.0, 0.8), 4)),
                    "reg_lambda": float(np.round(10 ** rng.uniform(-0.5, 1.3), 4)),
                }

            xgb_kwargs = {
                **candidate,
                "objective": "reg:absoluteerror" if alpha == 0.50 else "reg:quantileerror",
                "early_stopping_rounds": 25,
                "random_state": 42,
                "n_jobs": -1,
            }
            if alpha != 0.50:
                xgb_kwargs["quantile_alpha"] = alpha

            fold_losses = []
            best_iters = []
            for train_idx, val_idx in tscv.split(X):
                model = xgb.XGBRegressor(**xgb_kwargs)
                model.fit(
                    X[train_idx],
                    y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False,
                )
                preds = np.clip(model.predict(X[val_idx]), 0.1, 12.0)
                fold_losses.append(mean_pinball_loss(y[val_idx], preds, alpha=alpha))
                best_iters.append(int(getattr(model, "best_iteration", candidate["n_estimators"]) or 150))

            avg_loss = float(np.mean(fold_losses))
            opt_trees = max(80, int(np.mean(best_iters)))
            trial_record = {
                "trial": trial,
                "pinball_loss": round(avg_loss, 4),
                "optimal_n_estimators": opt_trees,
                **{k: v for k, v in candidate.items() if k != "n_estimators"},
            }
            self.tuning_history.append(trial_record)

            if avg_loss < best_score:
                best_score = avg_loss
                best_params = {**candidate, "n_estimators": opt_trees}
                if verbose:
                    print(
                        f"      [Trial {trial:02d}/{n_trials:02d}] ★ New Best CV Pinball Loss (α={alpha}): "
                        f"₹{best_score:.4f}/kWh | depth={best_params['max_depth']}, "
                        f"lr={best_params['learning_rate']}, trees={opt_trees}, "
                        f"min_child={best_params['min_child_weight']}, "
                        f"colsample={best_params['colsample_bytree']}"
                    )

        self.best_params = best_params
        self._init_models(best_params)
        return best_params, best_score

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        split_date: str = "2025-06-01",
        extra_features: List[str] | None = None,
        tune: bool = False,
        n_trials: int = 10,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Engineers features (automatically including Gujarat weather features if present in df
        plus any extra_features like Prophet/LSTM embeddings), optionally runs walk-forward
        TimeSeriesSplit hyperparameter tuning, trains all 4 XGBoost models, and returns
        the backtest DataFrame with metrics.
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

        cv_pinball_loss = None
        if tune:
            _, cv_pinball_loss = self.tune_hyperparameters(
                train_df,
                feature_cols=self.active_dam_features,
                target_col="DAM_MCP",
                alpha=0.50,
                n_trials=n_trials,
                verbose=True,
            )

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
        dam_p25_pinball = mean_pinball_loss(test_df["DAM_MCP"], test_df["Pred_DAM_P25"], alpha=0.25)
        dam_p50_pinball = mean_pinball_loss(test_df["DAM_MCP"], test_df["Pred_DAM_P50"], alpha=0.50)
        dam_p75_pinball = mean_pinball_loss(test_df["DAM_MCP"], test_df["Pred_DAM_P75"], alpha=0.75)
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
            "hyperparameters_tuned": tune,
            "best_hyperparameters": self.best_params,
            "cv_pinball_loss_p50": round(float(cv_pinball_loss), 4) if cv_pinball_loss is not None else None,
            "dam_mae_rs_kwh": round(float(dam_mae), 3),
            "dam_rmse_rs_kwh": round(float(dam_rmse), 3),
            "dam_pinball_p25": round(float(dam_p25_pinball), 4),
            "dam_pinball_p50": round(float(dam_p50_pinball), 4),
            "dam_pinball_p75": round(float(dam_p75_pinball), 4),
            "rtm_mae_rs_kwh": round(float(rtm_mae), 3),
            "p75_clearance_reliability_pct": round(float(p75_clearance), 2),
            "top_10_dam_feature_importances": importances,
        }
        return test_df, metrics
