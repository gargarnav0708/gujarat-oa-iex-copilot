"""
Gujarat Electricity Regulatory Commission (GERC) Tariff & Open Access Cost Stack Engine.
Covers HTP-I DISCOM Tariffs (DGVCL, PGVCL, UGVCL, MGVCL), Time-of-Use (ToU) Peak/Solar windows,
GETCO STU transmission charges, Cross-Subsidy Surcharge (CSS), Additional Surcharge (AS),
and technical network losses (ISTS, STU, and DISCOM Wheeling).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class OAMode(str, Enum):
    """Supported Gujarat Open Access regulatory and voltage connection profiles."""
    EHV_66KV_STANDARD = "66kV_Standard_OA"
    HT_11_33KV_STANDARD = "11_33kV_Standard_OA"
    HT_11_33KV_GREEN_CAPTIVE = "11_33kV_Green_Captive_Blend"


@dataclass
class ConsumerProfile:
    """Industrial C&I Consumer load and demand configuration."""
    name: str = "Gujarat 5 MW C&I Plant (Textile / Pharma / Steel)"
    base_load_mw: float = 4.0       # Continuous unshiftable base load (MW)
    flex_load_mw: float = 1.0       # Shiftable process/HVAC/pumping load (MW)
    oa_mode: OAMode = OAMode.EHV_66KV_STANDARD

    @property
    def total_avg_load_mw(self) -> float:
        return self.base_load_mw + self.flex_load_mw


@dataclass
class GERCConfig:
    """
    Official Gujarat Electricity Regulatory Commission (GERC) HTP-I & Open Access parameters
    (FY 2024-25 / FY 2025-26 baseline). All monetary values in Rs./kWh unless noted.
    """
    # 1. Default DISCOM Variable Tariff (HTP-I Category > 2500 kVA)
    discom_base_energy_charge: float = 4.45
    discom_fpppa_charge: float = 2.85
    # Time-of-Use (ToU) / Time-of-Day (ToD) Adjustments
    # Peak windows: 07:00-11:00 and 18:00-22:00 (+0.85 Rs/kWh)
    # Solar daytime window: 11:00-17:00 (-0.45 Rs/kWh)
    tou_peak_surcharge: float = 0.85
    tou_solar_rebate: float = -0.45

    # 2. Technical Transmission & Wheeling Losses (Fractional)
    ists_loss_pct: float = 0.0320             # 3.20% Central / Inter-State CTU loss
    stu_loss_pct: float = 0.0335              # 3.35% GETCO Intra-State STU loss
    dist_wheeling_loss_66kv: float = 0.0000   # 0.00% for 66kV+ EHV direct GETCO connection
    dist_wheeling_loss_11_33kv: float = 0.0750 # 7.50% for 11kV/33kV DISCOM network

    # 3. Open Access Per-Unit Statutory Charges & Surcharges (Rs./kWh delivered)
    stu_transmission_charge: float = 0.3909   # GETCO STOA rate (39.09 paise/kWh)
    wheeling_charge_66kv: float = 0.0000      # Exempt at 66kV EHV
    wheeling_charge_11_33kv: float = 0.6000   # Rs. 0.60/kWh for 11/33kV HT network
    cross_subsidy_surcharge: float = 1.7900   # GERC HTP-I CSS (Rs. 1.79/kWh)
    additional_surcharge: float = 1.3600      # GERC Additional Surcharge (Rs. 1.36/kWh)
    sldc_iex_trader_fees: float = 0.0900      # IEX (0.02) + Trader (0.04) + SLDC/NLDC (0.03)

    def get_discom_variable_tariff(self, hour_float: float) -> float:
        """Returns the effective variable DISCOM energy rate (Rs./kWh) for an hour [0, 24)."""
        base = self.discom_base_energy_charge + self.discom_fpppa_charge
        if (7.0 <= hour_float < 11.0) or (18.0 <= hour_float < 22.0):
            return round(base + self.tou_peak_surcharge, 4)
        elif 11.0 <= hour_float < 17.0:
            return round(base + self.tou_solar_rebate, 4)
        return round(base, 4)

    def get_loss_and_adder(self, mode: OAMode | str) -> tuple[float, float]:
        """
        Returns (net_delivery_factor, fixed_per_kwh_adder) for the specified OA mode.
        net_delivery_factor = (1 - L_ISTS) * (1 - L_STU) * (1 - L_Wheel)
        """
        mode_val = mode.value if isinstance(mode, OAMode) else str(mode)

        if mode_val == OAMode.EHV_66KV_STANDARD.value:
            loss_factor = (1.0 - self.ists_loss_pct) * (1.0 - self.stu_loss_pct)
            fixed_adder = (
                self.stu_transmission_charge
                + self.cross_subsidy_surcharge
                + self.additional_surcharge
                + self.sldc_iex_trader_fees
            )
        elif mode_val == OAMode.HT_11_33KV_STANDARD.value:
            loss_factor = (
                (1.0 - self.ists_loss_pct)
                * (1.0 - self.stu_loss_pct)
                * (1.0 - self.dist_wheeling_loss_11_33kv)
            )
            fixed_adder = (
                self.stu_transmission_charge
                + self.wheeling_charge_11_33kv
                + self.cross_subsidy_surcharge
                + self.additional_surcharge
                + self.sldc_iex_trader_fees
            )
        elif mode_val == OAMode.HT_11_33KV_GREEN_CAPTIVE.value:
            loss_factor = (
                (1.0 - self.ists_loss_pct)
                * (1.0 - self.stu_loss_pct)
                * (1.0 - self.dist_wheeling_loss_11_33kv)
            )
            fixed_adder = (
                self.stu_transmission_charge
                + (self.wheeling_charge_11_33kv * 0.5)
                + self.sldc_iex_trader_fees
            )
        else:
            raise ValueError(f"Unsupported OA mode: {mode_val}")

        return loss_factor, fixed_adder

    def get_oa_landed_cost(self, mcp_rs_kwh: float, mode: OAMode | str = OAMode.EHV_66KV_STANDARD) -> float:
        """
        Computes total landed cost at consumer periphery (Rs./kWh) for an IEX Market Clearing Price.
        """
        loss_factor, fixed_adder = self.get_loss_and_adder(mode)
        return (mcp_rs_kwh / loss_factor) + fixed_adder

    def get_breakeven_mcp(self, discom_rate: float, mode: OAMode | str = OAMode.EHV_66KV_STANDARD) -> float:
        """
        Computes the exact maximum IEX Market Clearing Price (Rs./kWh) where OA Landed Cost == DISCOM Rate.
        """
        loss_factor, fixed_adder = self.get_loss_and_adder(mode)
        return max(0.0, (discom_rate - fixed_adder) * loss_factor)

    def get_cost_stack_breakdown(self, mode: OAMode | str = OAMode.EHV_66KV_STANDARD) -> Dict[str, Any]:
        """Returns a dictionary summarizing the regulatory parameters for reporting."""
        loss_factor, fixed_adder = self.get_loss_and_adder(mode)
        return {
            "mode": mode.value if isinstance(mode, OAMode) else str(mode),
            "net_delivery_factor": round(loss_factor, 4),
            "effective_loss_pct": round((1.0 - loss_factor) * 100.0, 2),
            "fixed_oa_adder_rs_kwh": round(fixed_adder, 4),
            "breakeven_mcp_peak_rs_kwh": round(self.get_breakeven_mcp(self.get_discom_variable_tariff(8.0), mode), 2),
            "breakeven_mcp_normal_rs_kwh": round(self.get_breakeven_mcp(self.get_discom_variable_tariff(1.0), mode), 2),
            "breakeven_mcp_solar_rs_kwh": round(self.get_breakeven_mcp(self.get_discom_variable_tariff(13.0), mode), 2),
        }
