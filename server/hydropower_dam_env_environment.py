
import random
import math
import uuid
from typing import Optional

from models import HydropowerAction, HydropowerObservation, HydropowerState
from configs import TASKS


class HydropowerDamEnvEnvironment:
    """
    Hydropower Dam Management Environment.

    An agent operates a hydropower dam across a 90-day season (2160 hourly steps).
    Each step it decides how much water to release through the turbines.

    The core tension:
    - Release more now → more power revenue, but risks low reservoir later
    - Release less now → conserves water, but risks overflow during monsoon
    - Must always maintain minimum ecological flow downstream
    - Must never exceed flood threshold downstream

    State space (returned in observation):
    - reservoir_level: fraction of capacity (0.0 to 1.0)
    - inflow_forecast: next 3 hours of predicted inflow (m3/s)
    - grid_price: current electricity price (INR/kWh)
    - downstream_demand: minimum flow required (m3/s)

    Action space (discrete, 7 levels):
    - 0 → 0%   of max turbine capacity
    - 1 → 17%
    - 2 → 33%
    - 3 → 50%
    - 4 → 67%
    - 5 → 83%
    - 6 → 100%
    Spill is automatic if reservoir overflows — always penalized.
    """

    RELEASE_FRACTIONS = [0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0]

    TURBINE_EFFICIENCY = 0.85
    GRAVITY = 9.81
    WATER_DENSITY = 1000        # kg/m3
    EFFECTIVE_HEAD_M = 50       # metres
    TIMESTEP_HOURS = 1          # one step = one hour

    def __init__(self, task_id: str = "task_1"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(TASKS.keys())}")
        self.task_id = task_id
        self.cfg = TASKS[task_id]

        # Episode state — set properly in reset()
        self._episode_id: str = ""
        self._step: int = 0
        self._reservoir_m3: float = 0.0
        self._cumulative_reward: float = 0.0
        self._total_power_mwh: float = 0.0
        self._total_revenue_inr: float = 0.0          # NEW: raw revenue, no penalties
        self._flood_count: int = 0
        self._eco_violation_count: int = 0
        self._spill_count: int = 0
        self._steps_below_reservoir_threshold: int = 0  # NEW: steps below 30% capacity
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Required OpenEnv methods
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> HydropowerObservation:
        """Start a new 90-day episode."""
        if seed is not None:
            self._rng.seed(seed)
        else:
            self._rng.seed()

        self._episode_id = str(uuid.uuid4())[:8]
        self._step = 0
        self._cumulative_reward = 0.0
        self._total_power_mwh = 0.0
        self._total_revenue_inr = 0.0
        self._flood_count = 0
        self._eco_violation_count = 0
        self._spill_count = 0
        self._steps_below_reservoir_threshold = 0

        capacity = self.cfg["reservoir_capacity_m3"]
        self._reservoir_m3 = self.cfg["reservoir_init"] * capacity

        return HydropowerObservation(
            reservoir_level=self.cfg["reservoir_init"],
            inflow_forecast=self._get_inflow_forecast(),
            grid_price=self._get_grid_price(),
            downstream_demand=float(self.cfg["eco_min_flow_m3s"]),
            power_generated=0.0,
            flood_triggered=False,
            eco_violation=False,
            spill_triggered=False,
            reward=0.0,
            done=False,
            step=0,
            feedback=f"Episode started. Task: {self.cfg['name']}. Manage the dam across 90 days.",
        )

    def step(self, action: HydropowerAction) -> HydropowerObservation:
        """Execute one hourly timestep."""
        self._step += 1
        cfg = self.cfg
        weights = cfg["reward_weights"]
        capacity = cfg["reservoir_capacity_m3"]

        # --- 1. Compute actual release (m3/s) ---
        release_fraction = self.RELEASE_FRACTIONS[action.release_level]
        max_release = cfg["max_turbine_release_m3s"]
        turbine_release_m3s = release_fraction * max_release

        # --- 2. Get current inflow ---
        inflow_m3s = self._get_inflow_now()

        # --- 3. Update reservoir ---
        inflow_m3 = inflow_m3s * 3600
        release_m3 = turbine_release_m3s * 3600

        new_reservoir = self._reservoir_m3 + inflow_m3 - release_m3

        # --- 4. Check for overflow → auto spill ---
        spill_triggered = False
        if new_reservoir > capacity:
            spill_m3 = new_reservoir - capacity
            new_reservoir = capacity
            spill_triggered = True
            self._spill_count += 1
            turbine_release_m3s += spill_m3 / 3600

        # --- 5. Clamp reservoir to zero floor ---
        new_reservoir = max(0.0, new_reservoir)
        self._reservoir_m3 = new_reservoir
        reservoir_level = new_reservoir / capacity

        # --- 6. Track reservoir below threshold (30%) ---
        if reservoir_level < 0.30:
            self._steps_below_reservoir_threshold += 1

        # --- 7. Compute power and raw revenue ---
        power_w = (
            self.TURBINE_EFFICIENCY
            * self.WATER_DENSITY
            * self.GRAVITY
            * self.EFFECTIVE_HEAD_M
            * release_fraction * max_release
        )
        power_mw = power_w / 1e6
        grid_price = self._get_grid_price()
        revenue = power_mw * grid_price * self.TIMESTEP_HOURS  # INR

        self._total_power_mwh += power_mw * self.TIMESTEP_HOURS
        self._total_revenue_inr += revenue                      # NEW: track raw revenue

        # --- 8. Check violations ---
        flood_threshold = cfg["flood_threshold_m3s"]
        eco_min = cfg["eco_min_flow_m3s"]

        flood_triggered = turbine_release_m3s > flood_threshold
        eco_violation = turbine_release_m3s < eco_min

        if flood_triggered:
            self._flood_count += 1
        if eco_violation:
            self._eco_violation_count += 1

        # --- 9. Compute reward (revenue minus penalties) ---
        reward = revenue * weights["power_revenue"]

        if flood_triggered:
            reward -= weights["flood_penalty"] * (turbine_release_m3s - flood_threshold)

        if eco_violation:
            reward -= weights["eco_penalty"] * (eco_min - turbine_release_m3s)

        if reservoir_level < 0.2:
            reward -= weights["low_reservoir_penalty"] * (0.2 - reservoir_level) * 1000

        if spill_triggered:
            reward -= weights["spill_penalty"] * 100

        self._cumulative_reward += reward

        # --- 10. Check done ---
        done = self._step >= cfg["max_steps"]

        # --- 11. Build feedback message ---
        messages = []
        if flood_triggered:
            messages.append("FLOOD WARNING: release exceeded safe downstream threshold.")
        if eco_violation:
            messages.append("ECO VIOLATION: release below minimum ecological flow.")
        if spill_triggered:
            messages.append("SPILL: reservoir overflowed, emergency spill triggered.")
        if reservoir_level < 0.2:
            messages.append("LOW RESERVOIR: below 20% capacity.")
        if done:
            messages.append(
                f"Episode complete. "
                f"Total reward: {self._cumulative_reward:.1f} | "
                f"Power: {self._total_power_mwh:.1f} MWh | "
                f"Revenue: {self._total_revenue_inr:.0f} INR | "
                f"Floods: {self._flood_count} | "
                f"Eco violations: {self._eco_violation_count} | "
                f"Spills: {self._spill_count}."
            )
        feedback = " ".join(messages) if messages else None

        return HydropowerObservation(
            reservoir_level=reservoir_level,
            inflow_forecast=self._get_inflow_forecast(),
            grid_price=grid_price,
            downstream_demand=float(eco_min),
            power_generated=power_mw,
            flood_triggered=flood_triggered,
            eco_violation=eco_violation,
            spill_triggered=spill_triggered,
            reward=reward,
            done=done,
            step=self._step,
            feedback=feedback,
        )

    @property
    def state(self) -> HydropowerState:
        """Current episode metadata."""
        return HydropowerState(
            episode_id=self._episode_id,
            total_steps=self._step,
            cumulative_reward=self._cumulative_reward,
            total_power_generated=self._total_power_mwh,
            flood_count=self._flood_count,
            eco_violation_count=self._eco_violation_count,
            spill_count=self._spill_count,
            season_day=self._step // 24,
        )

    def episode_stats(self) -> dict:
        """
        Full stats dict for the grader.
        Call this after episode ends (done=True).
        """
        return {
            "total_steps": self._step,
            "flood_count": self._flood_count,
            "eco_violation_count": self._eco_violation_count,
            "spill_count": self._spill_count,
            "total_power_mwh": self._total_power_mwh,
            "total_revenue_inr": self._total_revenue_inr,
            "steps_below_reservoir_threshold": self._steps_below_reservoir_threshold,
            "cumulative_reward": self._cumulative_reward,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_inflow_now(self) -> float:
        """
        Sample current inflow (m3/s).
        Seasonal pattern: inflow peaks at mid-season for monsoon,
        stays low throughout for dry, transitions for mixed.
        """
        cfg = self.cfg["inflow"]
        season = self.cfg["season"]
        progress = self._step / self.cfg["max_steps"]

        if season == "dry":
            seasonal_factor = 1.0
        elif season == "monsoon":
            seasonal_factor = 1.0 + math.sin(progress * math.pi)
        else:  # mixed
            seasonal_factor = 1.0 + max(0.0, math.sin((progress - 0.4) * math.pi))

        base = cfg["base_m3s"] * seasonal_factor
        noise = self._rng.gauss(0, cfg["noise_std"])
        return max(0.0, base + cfg["seasonal_amplitude"] * seasonal_factor * 0.3 + noise)

    def _get_inflow_forecast(self) -> list:
        """Return noisy 3-step ahead inflow forecast."""
        return [
            max(0.0, self._get_inflow_now() + self._rng.gauss(0, self.cfg["inflow"]["noise_std"] * 0.5))
            for _ in range(3)
        ]

    def _get_grid_price(self) -> float:
        """
        Grid price follows a daily cycle with morning and evening peaks.
        """
        cfg = self.cfg["grid_price"]
        hour_of_day = self._step % 24
        daily_cycle = (
            math.sin((hour_of_day - 8) * math.pi / 12) +
            math.sin((hour_of_day - 19) * math.pi / 6) * 0.5
        )
        noise = self._rng.gauss(0, cfg["noise_std"])
        price = cfg["base_inr_kwh"] + cfg["daily_amplitude"] * daily_cycle + noise
        return max(1.0, price)
