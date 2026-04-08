
from typing import Optional, List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class HydropowerAction(Action):
    """What the agent sends each timestep."""

    release_level: int = Field(
        ...,
        description="Discrete release level: 0=0%, 1=17%, 2=33%, 3=50%, 4=67%, 5=83%, 6=100% of max turbine capacity"
    )


class HydropowerObservation(Observation):
    """What the environment returns after each step."""

    reservoir_level: float = Field(
        ...,
        description="Current reservoir level as fraction of capacity (0.0 to 1.0)"
    )
    inflow_forecast: List[float] = Field(
        default_factory=list,
        description="Inflow forecast for next 3 timesteps (m3/s)"
    )
    grid_price: float = Field(
        ...,
        description="Current electricity grid price (INR/kWh)"
    )
    downstream_demand: float = Field(
        ...,
        description="Minimum ecological/agricultural flow required downstream (m3/s)"
    )
    power_generated: float = Field(
        default=0.0,
        description="Power generated this step (MW)"
    )
    flood_triggered: bool = Field(
        default=False,
        description="True if downstream flood threshold was breached this step"
    )
    eco_violation: bool = Field(
        default=False,
        description="True if ecological minimum flow was violated this step"
    )
    spill_triggered: bool = Field(
        default=False,
        description="True if emergency spill was auto-triggered due to overflow"
    )
    reward: float = Field(
        default=0.0,
        description="Step reward"
    )
    done: bool = Field(
        default=False,
        description="True when episode is over"
    )
    step: int = Field(
        default=0,
        description="Current timestep within episode"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Human-readable message about what happened this step"
    )


class HydropowerState(State):
    """Episode-level metadata."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    total_steps: int = Field(default=0, description="Total steps taken so far")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    total_power_generated: float = Field(default=0.0, description="Total MWh generated")
    flood_count: int = Field(default=0, description="Number of flood events this episode")
    eco_violation_count: int = Field(default=0, description="Number of ecological violations")
    spill_count: int = Field(default=0, description="Number of emergency spills")
    season_day: int = Field(default=0, description="Current day within the 90-day season")
