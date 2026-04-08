from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from hydropower_dam_env_environment import HydropowerDamEnvEnvironment
from models import HydropowerAction, HydropowerObservation, HydropowerState

app: FastAPI = create_app(
    env=HydropowerDamEnvEnvironment,
    action_cls=HydropowerAction,
    observation_cls=HydropowerObservation,
    max_concurrent_envs=1,
)
