
"""
Three tasks for the Hydropower Dam Management environment.

Each task has:
- A concrete objective the agent must accomplish
- A programmatic grader that scores 0.0 to 1.0
- Clear difficulty progression: easy -> medium -> hard

One episode = one operational season = 90 days at hourly timesteps = 2160 steps.
"""

TASKS = {

    "task_1": {
        "task_id": "task_1",
        "difficulty": "easy",
        "name": "Dry Season Stability",
        "description": (
            "Manage the dam through a dry season. Inflow is low and predictable. "
            "Your objective is to maintain minimum ecological flow at all times "
            "and keep the reservoir above 30% capacity throughout. "
            "There is no flood risk this season — focus on conservation."
        ),
        "objective": "Maintain eco flow and reservoir level above 30% for 90 days.",
        "max_steps": 2160,
        "reservoir_init": 0.6,
        "reservoir_capacity_m3": 1e8,
        "max_turbine_release_m3s": 300,
        "flood_threshold_m3s": 400,
        "eco_min_flow_m3s": 50,
        "season": "dry",
        "inflow": {
            "base_m3s": 80,
            "seasonal_amplitude": 20,
            "noise_std": 10,
        },
        "grid_price": {
            "base_inr_kwh": 4.0,
            "daily_amplitude": 1.0,
            "noise_std": 0.2,
        },
        "reward_weights": {
            "power_revenue": 1.0,
            "flood_penalty": 5.0,
            "eco_penalty": 3.0,
            "low_reservoir_penalty": 4.0,
            "spill_penalty": 0.5,
        },
        "grader": {
            "eco_compliance_weight": 0.5,
            "reservoir_stability_weight": 0.5,
            "reservoir_stability_threshold": 0.30,
        },
    },

    "task_2": {
        "task_id": "task_2",
        "difficulty": "medium",
        "name": "Monsoon Flood Control",
        "description": (
            "Manage the dam through a high-inflow monsoon season. "
            "Inflow is large and unpredictable — the reservoir will fill fast. "
            "Your objective is to release water aggressively enough to prevent flooding "
            "while still generating a minimum power target. "
            "Every flood event will cost you score."
        ),
        "objective": "Zero flood events. Generate at least 8000 MWh over the season.",
        "max_steps": 2160,
        "reservoir_init": 0.5,
        "reservoir_capacity_m3": 1e8,
        "max_turbine_release_m3s": 300,
        "flood_threshold_m3s": 400,
        "eco_min_flow_m3s": 50,
        "season": "monsoon",
        "inflow": {
            "base_m3s": 250,
            "seasonal_amplitude": 80,
            "noise_std": 40,
        },
        "grid_price": {
            "base_inr_kwh": 4.5,
            "daily_amplitude": 2.0,
            "noise_std": 0.5,
        },
        "reward_weights": {
            "power_revenue": 1.0,
            "flood_penalty": 10.0,
            "eco_penalty": 3.0,
            "low_reservoir_penalty": 1.0,
            "spill_penalty": 1.0,
        },
        "grader": {
            "flood_free_weight": 0.6,
            "power_weight": 0.4,
            "power_target_mwh": 8000,
        },
    },

    "task_3": {
        "task_id": "task_3",
        "difficulty": "hard",
        "name": "Price-Aware Dispatch",
        "description": (
            "Manage the dam across a mixed season — starts dry, monsoon arrives halfway. "
            "Grid prices are highly volatile with strong daily peaks. "
            "Your objective is to maximise revenue by timing releases to price peaks, "
            "while keeping flood events below 5% of steps and eco violations below 5% of steps. "
            "A fixed-rate release strategy will score poorly here."
        ),
        "objective": "Maximise revenue. Keep floods and eco violations each under 5% of steps.",
        "max_steps": 2160,
        "reservoir_init": 0.4,
        "reservoir_capacity_m3": 1e8,
        "max_turbine_release_m3s": 300,
        "flood_threshold_m3s": 400,
        "eco_min_flow_m3s": 50,
        "season": "mixed",
        "inflow": {
            "base_m3s": 120,
            "seasonal_amplitude": 150,
            "noise_std": 60,
        },
        "grid_price": {
            "base_inr_kwh": 5.0,
            "daily_amplitude": 3.0,
            "noise_std": 1.0,
        },
        "reward_weights": {
            "power_revenue": 1.0,
            "flood_penalty": 8.0,
            "eco_penalty": 3.0,
            "low_reservoir_penalty": 2.0,
            "spill_penalty": 1.5,
        },
        "grader": {
            "revenue_weight": 0.5,
            "flood_compliance_weight": 0.3,
            "eco_compliance_weight": 0.2,
            "flood_compliance_threshold": 0.95,
            "eco_compliance_threshold": 0.95,
            "theoretical_max_revenue": 150000,
        },
    },

}
