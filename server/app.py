
from fastapi import FastAPI
from openenv.core.env_server.base_app import create_app
from hydropower_dam_env_environment import HydropowerDamEnvEnvironment
from models import HydropowerAction, HydropowerObservation, HydropowerState

app: FastAPI = create_app(
    environment_class=HydropowerDamEnvEnvironment,
    action_class=HydropowerAction,
    observation_class=HydropowerObservation,
    state_class=HydropowerState,
)
