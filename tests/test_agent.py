"""Model-selection tests for build_agent. No network or credentials needed."""

import pytest

from weather_agent.agent import build_agent


def test_gemini_flash_runs_with_thinking_off():
    agent = build_agent("gemini-2.5-flash")

    assert agent.model == "gemini-2.5-flash"
    assert agent.generate_content_config.thinking_config.thinking_budget == 0


def test_other_gemini_models_keep_default_config():
    assert build_agent("gemini-2.5-pro").generate_content_config is None


def test_claude_uses_anthropic_api_when_key_set(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    agent = build_agent("claude-opus-5-5")

    assert type(agent.model).__name__ == "AnthropicLlm"
    assert agent.model.model == "claude-opus-5-5"
    assert agent.generate_content_config.effort == "low"


def test_claude_uses_agent_platform_without_key(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    agent = build_agent("claude-opus-5-5")

    assert type(agent.model).__name__ == "Claude"
