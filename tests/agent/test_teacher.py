"""Tests for the post-run teacher agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.providers.base import LLMResponse, ToolCallRequest


@pytest.mark.asyncio
async def test_teacher_updates_learn_file(tmp_path) -> None:
    from nanobot.agent.teacher import TeacherAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "TEACHER.md").write_text("Review the trace and update LEARN.md.", encoding="utf-8")
    (workspace / "LEARN.md").write_text("# Learned Improvements\n\n- Existing rule.\n", encoding="utf-8")

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            id="call_1",
            name="save_learning_item",
            arguments={
                "learning_item": "- New rule.",
                "summary": "added new rule",
            },
        )],
    ))

    teacher = TeacherAgent(workspace=workspace, provider=provider, model="test-model")
    changed = await teacher.review_turn(
        turn_messages=[
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I will inspect the file first."},
        ],
        channel="cli",
        chat_id="direct",
    )

    assert changed is True
    assert (workspace / "LEARN.md").read_text(encoding="utf-8") == (
        "# Learned Improvements\n\n- Existing rule.\n- New rule.\n"
    )


@pytest.mark.asyncio
async def test_teacher_keeps_learn_file_unchanged_when_no_item(tmp_path) -> None:
    from nanobot.agent.teacher import TeacherAgent

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "TEACHER.md").write_text("Review the trace and update LEARN.md.", encoding="utf-8")
    original = "# Learned Improvements\n\n- Existing rule.\n"
    (workspace / "LEARN.md").write_text(original, encoding="utf-8")

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(
            id="call_1",
            name="save_learning_item",
            arguments={
                "learning_item": "",
                "summary": "unchanged",
            },
        )],
    ))

    teacher = TeacherAgent(workspace=workspace, provider=provider, model="test-model")
    changed = await teacher.review_turn(
        turn_messages=[{"role": "assistant", "content": "done"}],
        channel="cli",
        chat_id="direct",
    )

    assert changed is True
    assert (workspace / "LEARN.md").read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_teacher_skips_when_prompt_file_missing(tmp_path) -> None:
    from nanobot.agent.teacher import TeacherAgent

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    teacher = TeacherAgent(workspace=tmp_path, provider=provider, model="test-model")

    changed = await teacher.review_turn(turn_messages=[{"role": "user", "content": "hello"}])

    assert changed is False
    provider.chat_with_retry.assert_not_awaited()
