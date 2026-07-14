"""RebatePivotalityAnalyzer: exchange incentive optimization framework (Issue #10).

Ingests standardized P&L metrics from Issue #9 and computes the exact exchange 
incentives required to maintain a break-even P&L. Evaluates fee waivers, volume-tiered 
rebates, fixed-cost subsidies, and cross-references against real-world CME programs.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

class RebatePivotalityAnalyzer:
    """Executes the exchange rebate pivotality analysis and feasibility check."""

    def __init__(self, current_passive_fee: float, current_aggressive_fee: float, daily_fixed_costs: float = 0.0):
        """
        Initializes the analyzer with baseline fees and infrastructure costs.
        (Negative passive_fee denotes a per-contract rebate).
        """
        self.current_passive_fee = current_passive_fee
        self.current_aggressive_fee = current_aggressive_fee
        self.daily_fixed_costs = daily_fixed_costs
        
        # Real-world CME incentive reference benchmarks (Representative examples for Issue #10)
        self.cme_programs = {
            "Standard Member": {"passive_fee": 1.25, "data_subsidy": 0.0},
            "Base Market Maker (Fee Discount)": {"passive_fee": 0.40, "data_subsidy": 0.0},
            "Designated Market Maker (DMM)": {"passive_fee": 0.00, "data_subsidy": 0.0},
            "Top-Tier DMM (Rebate + Data Subsidy)": {"passive_fee": -0.15, "data_subsidy": self.daily_fixed_costs}
        }

        # Example Volume-Tiered Rebate Schedule (Threshold: Rate)
        self.volume_tiers = {
            0: 0.40,       # 0 to 999 contracts: Pay $0.40
            1000: 0.00,    # 1000 to 4999 contracts: Free
            5000: -0.20    # 5000+ contracts: Earn $0.20 rebate
        }

    def _get_average_metrics(self, pnl_samples: List[pd.Series]) -> pd.Series:
        """Helper to average the Bernoulli-driven simulation seeds from Issue #9."""
        return pd.DataFrame(pnl_samples).mean()

    def _calculate_effective_tiered_rate(self, total_volume: float) -> float:
        """Calculates the blended per-contract rate based on a volume-tiered schedule."""
        sorted_thresholds = sorted(self.volume_tiers.keys())
        applicable_rate = self.volume_tiers[0]
        
        for threshold in sorted_thresholds:
            if total_volume >= threshold:
                applicable_rate = self.volume_tiers[threshold]
            else:
                break
        return applicable_rate

    def generate_pivotality_report(
        self, 
        standard_pnl_samples: List[pd.Series], 
        dmm_pnl_samples: Optional[List[pd.Series]] = None
    ) -> pd.DataFrame:
        """
        Generates the comparative pivotality report detailing required rebate levels.
        Optionally ingests DMM priority-matching samples (higher queue-head probability) 
        to evaluate allocation benefits side-by-side.
        """
        results: List[Dict[str, Any]] = []
        
        # 1. Process Standard Market Maker Flow
        std_avg = self._get_average_metrics(standard_pnl_samples)
        self._append_scenarios(results, std_avg, flow_type="Standard Flow")

        # 2. Process DMM Priority Matching Flow (if provided)
        if dmm_pnl_samples is not None:
            dmm_avg = self._get_average_metrics(dmm_pnl_samples)
            self._append_scenarios(results, dmm_avg, flow_type="DMM Priority Matching")

        report_df = pd.DataFrame(results)
        for col in ["Passive Fee/Rebate", "Expected Net P&L", "Gross Trading P&L"]:
            report_df[col] = report_df[col].round(4)
            
        return report_df

    def _append_scenarios(self, results: List[Dict[str, Any]], avg_metrics: pd.Series, flow_type: str):
        """Computes all support mechanisms and solves for break-even targets."""
        filled = avg_metrics["filled_contracts"]
        hedged = avg_metrics["hedged_contracts"]
        
        # Isolate Gross Trading PnL (pre-exchange fees and pre-fixed costs)
        pure_legging_costs = avg_metrics["hedge_costs"] - (self.current_aggressive_fee * hedged)
        gross_pnl = avg_metrics["cash"] + avg_metrics["position_mark"] - pure_legging_costs

        # --- Scenario A: Status Quo ---
        sq_net = gross_pnl - (self.current_passive_fee * filled) - (self.current_aggressive_fee * hedged) - self.daily_fixed_costs
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Status Quo",
            "Support Mechanism": "None",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": self.current_passive_fee,
            "Expected Net P&L": sq_net
        })

        # --- Scenario B: Total Fee Waiver ---
        waiver_net = gross_pnl - self.daily_fixed_costs
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Total Fee Waiver",
            "Support Mechanism": "Zero Trans. Fees",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": 0.0,
            "Expected Net P&L": waiver_net
        })

        # --- Scenario C: Volume-Tiered Rebate ---
        blended_rate = self._calculate_effective_tiered_rate(filled)
        tiered_net = gross_pnl - (blended_rate * filled) - (self.current_aggressive_fee * hedged) - self.daily_fixed_costs
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Volume-Tiered Check",
            "Support Mechanism": "Tiered Rebate Schedule",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": blended_rate,
            "Expected Net P&L": tiered_net
        })

        # --- Scenario D: Exact Break-Even Solver (No Subsidy) ---
        be_fee = (gross_pnl - (self.current_aggressive_fee * hedged) - self.daily_fixed_costs) / filled if filled > 0 else 0.0
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Target Break-Even",
            "Support Mechanism": "Rebate Only",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": be_fee,
            "Expected Net P&L": 0.0
        })

        # --- Scenario E: Break-Even Solver (With Data/Messaging Subsidy) ---
        be_subsidized = (gross_pnl - (self.current_aggressive_fee * hedged)) / filled if filled > 0 else 0.0
        results.append({
            "Flow Type": flow_type,
            "Scenario": "Subsidized Break-Even",
            "Support Mechanism": "Data Subsidy + Rebate",
            "Gross Trading P&L": gross_pnl,
            "Passive Fee/Rebate": be_subsidized,
            "Expected Net P&L": 0.0
        })

    def evaluate_cme_feasibility(self, required_break_even_fee: float, fixed_costs_subsidized: bool = False) -> pd.DataFrame:
        """
        Cross-references the calculated break-even incentives against publicly 
        available CME programs to evaluate commercial feasibility.
        """
        feasibility = []
        for program_name, params in self.cme_programs.items():
            prog_fee = params["passive_fee"]
            subsidy = params["data_subsidy"]
            
            # Lower fee (or more negative rebate) is financially superior
            if fixed_costs_subsidized and subsidy >= self.daily_fixed_costs:
                is_feasible = prog_fee <= required_break_even_fee 
            elif not fixed_costs_subsidized:
                is_feasible = prog_fee <= required_break_even_fee
            else:
                is_feasible = False
                
            feasibility.append({
                "CME Program": program_name,
                "Program Passive Fee": prog_fee,
                "Data/Messaging Subsidy": "Included" if subsidy > 0 else "None",
                "Commercially Feasible?": "Yes" if is_feasible else "No - Insufficient Edge"
            })
            
        return pd.DataFrame(feasibility)