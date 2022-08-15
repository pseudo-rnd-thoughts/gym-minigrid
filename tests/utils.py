"""Finds all the specs that we can test with"""
import gym

all_testing_env_specs = [
    env_spec
    for env_spec in gym.envs.registry.values()
    if env_spec.entry_point.startswith("gym_minigrid.envs")
]
