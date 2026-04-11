import pytest
from environment.env import VoiceAuthenticityEnv, TASKS

def test_reset_returns_observation():
    env = VoiceAuthenticityEnv("clean_detection")
    obs = env.reset()
    assert obs is not None
    assert obs.step_number == 0
    assert obs.task_name == "clean_detection"
    assert "hint" in obs.dict()

def test_step_returns_reward_in_range():
    env = VoiceAuthenticityEnv("clean_detection")
    env.reset()
    obs, reward, done, info = env.step({"action_type": "request_temporal_features"})
    assert 0.05 <= reward <= 0.95
    assert not done

def test_five_actions_complete_episode():
    env = VoiceAuthenticityEnv("clean_detection")
    env.reset()
    actions = [
        "request_temporal_features",
        "request_spectral_features",
        "request_comparison",
        "analyze_evidence",
        "final_classify"
    ]
    
    for i, act in enumerate(actions):
        obs, reward, done, info = env.step({
            "action_type": act,
            "label": 0,
            "confidence": 0.8,
            "reasoning": "Test reasoning"
        })
        if i < len(actions) - 1:
            assert not done
        else:
            assert done

def test_reward_never_zero_or_one():
    env = VoiceAuthenticityEnv("clean_detection")
    env.reset()
    # Test an action that could get penalties or rewards
    obs, reward, done, info = env.step({"action_type": "request_temporal_features"})
    assert reward != 0.0
    assert reward != 1.0

def test_all_five_tasks_load():
    for task in TASKS:
        env = VoiceAuthenticityEnv(task)
        assert env.task_name == task
        obs = env.reset()
        assert obs.task_name == task
