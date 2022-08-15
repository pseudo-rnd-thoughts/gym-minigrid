import math

import gym
import numpy as np
import pytest
from gym.utils.env_checker import data_equivalence

from gym_minigrid.envs import EmptyEnv
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import (
    ActionBonusWrapper,
    DictObservationSpaceWrapper,
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    OneHotPartialObsWrapper,
    ReseedWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    StateBonusWrapper,
    SymbolicObsWrapper,
    ViewSizeWrapper,
)
from tests.utils import all_testing_env_specs

SEEDS = [100, 243, 500]
NUM_STEPS = 100


@pytest.mark.parametrize(
    "wrapper",
    [
        ActionBonusWrapper,
        ReseedWrapper,
        ImgObsWrapper,
        FlatObsWrapper,
        ViewSizeWrapper,
        DictObservationSpaceWrapper,
        OneHotPartialObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
        SymbolicObsWrapper,
        # DirectionObsWrapper, todo re-add once fixed
        ActionBonusWrapper,
    ],
)
@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_all_wrappers(wrapper, env_spec):
    _env = env_spec.make(new_step_api=True)
    env = wrapper(_env)

    obs = env.reset()
    assert obs in env.observation_space
    obs, info = env.reset(return_info=True)
    assert obs in env.observation_space
    obs = env.reset(seed=0)
    assert obs in env.observation_space

    for _ in range(10):
        obs, _, _, terminated, truncated = env.step(env.action_space.sample())
        assert obs in env.observation_space

        if terminated or truncated:
            break

    env.close()


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_reseed_wrapper(env_spec):
    """
    Test the ReseedWrapper with a list of SEEDS.
    """
    unwrapped_env = env_spec.make(new_step_api=True)
    env = env_spec.make(new_step_api=True)
    env = ReseedWrapper(env, seeds=SEEDS)
    env.action_space.seed(0)

    for seed in SEEDS:
        env.reset()
        unwrapped_env.reset(seed=seed)
        for time_step in range(NUM_STEPS):
            action = env.action_space.sample()

            env_step_results = env.step(action)
            unwrapped_env_step_results = unwrapped_env.step(action)

            data_equivalence(env_step_results, unwrapped_env_step_results)

            assert env_step_results[0] in env.observation_space
            assert unwrapped_env_step_results[0] in unwrapped_env.observation_space

            # Start the next seed
            if env_step_results[2] or env_step_results[3]:
                break

    env.close()
    unwrapped_env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_state_bonus_wrapper(env_id):
    env = gym.make(env_id, new_step_api=True)
    wrapped_env = StateBonusWrapper(gym.make(env_id, new_step_api=True))

    for _ in range(10):
        wrapped_env.reset()
        for _ in range(5):
            wrapped_env.step(MiniGridEnv.Actions.forward)

    # Turn lef 3 times (check that actions don't influence bonus)
    wrapped_rew = 0
    for _ in range(3):
        _, wrapped_rew, _, _, _ = wrapped_env.step(MiniGridEnv.Actions.left)

    env.reset()
    for _ in range(5):
        env.step(MiniGridEnv.Actions.forward)

    # Turn right 3 times
    rew = 0
    for _ in range(3):
        _, rew, _, _, _ = env.step(MiniGridEnv.Actions.right)

    expected_bonus_reward = rew + 1 / math.sqrt(13)
    assert expected_bonus_reward == wrapped_rew

    env.close()
    wrapped_env.close()


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_dict_observation_space_wrapper(env_spec):
    env = env_spec.make(new_step_api=True)
    env = DictObservationSpaceWrapper(env)

    obs = env.reset()
    assert obs in env.observation_space
    assert env.string_to_indices(env.mission) == [
        value for value in obs["mission"] if value > 0
    ]

    obs, _, _, _, _ = env.step(0)
    assert env.string_to_indices(env.mission) == [
        value for value in obs["mission"] if value > 0
    ]

    env.close()


class EmptyEnvWithExtraObs(EmptyEnv):
    """
    Custom environment with an extra observation
    """

    def __init__(self):
        super().__init__(size=5)
        self.observation_space["size"] = gym.spaces.Box(
            low=0, high=np.iinfo(np.uint).max, shape=(2,), dtype=np.uint
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs["size"] = np.array([self.width, self.height], dtype=np.uint)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs["size"] = np.array([self.width, self.height], dtype=np.uint)
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize(
    "wrapper",
    [
        OneHotPartialObsWrapper,
        RGBImgObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
def test_agent_sees_method(wrapper):
    env1 = wrapper(EmptyEnvWithExtraObs())
    env2 = wrapper(gym.make("MiniGrid-Empty-5x5-v0", new_step_api=True))

    obs1 = env1.reset(seed=0)
    obs2 = env2.reset(seed=0)
    assert obs1 in env1.observation_space
    assert obs2 in env2.observation_space
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

    obs1, reward1, terminated1, truncated1, _ = env1.step(0)
    obs2, reward2, terminated2, truncated2, _ = env2.step(0)
    assert obs1 in env1.observation_space
    assert obs2 in env2.observation_space
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

    env1.close()
    env2.close()
