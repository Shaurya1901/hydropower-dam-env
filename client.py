
from typing import Optional
from openenv.core.client import BaseEnvClient
from models import HydropowerAction, HydropowerObservation, HydropowerState


class HydropowerDamEnvClient(BaseEnvClient):
    """
    Client for connecting to the Hydropower Dam Management environment.

    Usage:
        client = HydropowerDamEnvClient(host="localhost", port=8000)
        obs = client.reset()
        obs = client.step(HydropowerAction(release_level=3))
    """

    action_class = HydropowerAction
    observation_class = HydropowerObservation
    state_class = HydropowerState

    def reset(self, seed: Optional[int] = None) -> HydropowerObservation:
        """Start a new episode."""
        return super().reset(seed=seed)

    def step(self, action: HydropowerAction) -> HydropowerObservation:
        """Send a release action and get back the next observation."""
        return super().step(action)

    def state(self) -> HydropowerState:
        """Get current episode metadata."""
        return super().state()
