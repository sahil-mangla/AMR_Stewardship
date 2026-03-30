import pytest
import sys, os

# Ensure the root and env are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.asp_env import ASPEnv

@pytest.fixture
def env():
    """Returns a fresh ASP environment."""
    return ASPEnv(task_id="dynamic")

@pytest.fixture
def reset_obs(env):
    """Returns the observation from a fresh reset."""
    obs, info = env.reset(seed=42)
    return obs, info
