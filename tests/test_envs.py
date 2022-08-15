import warnings

import gym
import pytest
from gym.envs.registration import EnvSpec
from gym.utils.env_checker import check_env, data_equivalence

from gym_minigrid.minigrid import Grid, MissionSpace
from tests.utils import all_testing_env_specs


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_env_checker(spec):
    # Capture warnings
    env = spec.make(disable_env_checker=True).unwrapped

    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as check_env_warnings:
        warnings.simplefilter("always")
        check_env(env)

    assert len(check_env_warnings) == 0


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[env.id for env in all_testing_env_specs]
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, terminated, truncated and info are equals between the two envs
    """
    env_1 = env_spec.make(disable_env_checker=True, new_step_api=True)
    env_2 = env_spec.make(disable_env_checker=True, new_step_api=True)

    initial_obs_1 = env_1.reset(seed=SEED)
    initial_obs_2 = env_2.reset(seed=SEED)
    assert initial_obs_2 in env_1.observation_space
    data_equivalence(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = env_2.step(action)

        data_equivalence(obs_1, obs_2)
        assert env_1.observation_space.contains(
            obs_1
        )  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert (
            terminated_1 == terminated_2
        ), f"[{time_step}] terminated 1={terminated_1}, terminated 2={terminated_2}"
        assert (
            truncated_1 == truncated_2
        ), f"[{time_step}] truncated 1={truncated_1}, truncated 2={truncated_2}"
        data_equivalence(info_1, info_2)

        if (
            terminated_1 or truncated_1
        ):  # terminated_2 and truncated_2 verified by previous assertion
            break

    env_1.close()
    env_2.close()


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_render_modes(spec):
    env = spec.make(new_step_api=True)

    for mode in env.metadata.get("render_modes", []):
        if mode != "human":
            new_env = spec.make(new_step_api=True)

            new_env.reset()
            new_env.step(new_env.action_space.sample())
            new_env.render(mode=mode)
            new_env.close()
    env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-DoorKey-6x6-v0"])
def test_agent_sees_method(env_id):
    env = gym.make(env_id, new_step_api=True)
    goal_pos = (env.grid.width - 2, env.grid.height - 2)

    # Test the "in" operator on grid objects
    assert ("green", "goal") in env.grid
    assert ("blue", "key") not in env.grid

    # Test the env.agent_sees() function
    env.reset()
    for i in range(0, 500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        grid, _ = Grid.decode(obs["image"])
        goal_visible = ("green", "goal") in grid

        agent_sees_goal = env.agent_sees(*goal_pos)
        assert agent_sees_goal == goal_visible
        if terminated or truncated:
            env.reset()

    env.close()


def test_mission_space():
    # Test placeholders
    mission_space = MissionSpace(
        mission_func=lambda color, obj_type: f"Get the {color} {obj_type}.",
        ordered_placeholders=[["green", "red"], ["ball", "key"]],
    )

    assert mission_space.contains("Get the green ball.")
    assert mission_space.contains("Get the red key.")
    assert not mission_space.contains("Get the purple box.")

    # Test passing inverted placeholders
    assert not mission_space.contains("Get the key red.")

    # Test passing extra repeated placeholders
    assert not mission_space.contains("Get the key red key.")

    # Test contained placeholders like "get the" and "go get the". "get the" string is contained in both placeholders.
    mission_space = MissionSpace(
        mission_func=lambda get_syntax, obj_type: f"{get_syntax} {obj_type}.",
        ordered_placeholders=[
            ["go get the", "get the", "go fetch the", "fetch the"],
            ["ball", "key"],
        ],
    )

    assert mission_space.contains("get the ball.")
    assert mission_space.contains("go get the key.")
    assert mission_space.contains("go fetch the ball.")

    # Test repeated placeholders
    mission_space = MissionSpace(
        mission_func=lambda get_syntax, color_1, obj_type_1, color_2, obj_type_2: f"{get_syntax} {color_1} {obj_type_1} and the {color_2} {obj_type_2}.",
        ordered_placeholders=[
            ["go get the", "get the", "go fetch the", "fetch the"],
            ["green", "red"],
            ["ball", "key"],
            ["green", "red"],
            ["ball", "key"],
        ],
    )

    assert mission_space.contains("get the green key and the green key.")
    assert mission_space.contains("go fetch the red ball and the green key.")
