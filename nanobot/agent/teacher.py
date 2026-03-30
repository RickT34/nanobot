"""Post-run teacher agent for self-improvement notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import current_time_str

_SAVE_LEARNINGS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_learning_item",
            "description": "Persist at most one new learning item for future agent runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learning_item": {
                        "type": "string",
                        "description": "One markdown list item to append to LEARN.md, or an empty string if nothing should be added.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the review result.",
                    },
                },
                "required": ["learning_item", "summary"],
            },
        },
    }
]

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    text = (content or "").lower()
    return any(marker in text for marker in _TOOL_CHOICE_ERROR_MARKERS)


def _normalize_save_learnings_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def _stringify_block(block: Any) -> str:
    if isinstance(block, dict):
        if block.get("type") == "text":
            return str(block.get("text", ""))
        if block.get("type") == "image_url":
            path = (block.get("_meta") or {}).get("path", "")
            return f"[image: {path}]" if path else "[image]"
    if isinstance(block, (dict, list)):
        return json.dumps(block, ensure_ascii=False)
    return str(block)


class TeacherAgent:
    """Review completed runs and update LEARN.md with actionable lessons."""

    _MAX_MESSAGE_CHARS = 4000

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        timezone: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.timezone = timezone
        self.teacher_file = workspace / "TEACHER.md"
        self.learn_file = workspace / "LEARN.md"

    def is_enabled(self) -> bool:
        return self.teacher_file.exists()

    def _read_teacher_prompt(self) -> str:
        if self.teacher_file.exists():
            return self.teacher_file.read_text(encoding="utf-8")
        return ""

    def _append_learning_item(self, item: str) -> None:
        existing = self.learn_file.read_text(encoding="utf-8") if self.learn_file.exists() else ""
        base = existing.rstrip()
        if base:
            updated = f"{base}\n{item}\n"
        else:
            updated = f"# Learned Improvements\n\n{item}\n"
        self.learn_file.write_text(updated, encoding="utf-8")

    @classmethod
    def _format_messages(cls, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, message in enumerate(messages, start=1):
            role = str(message.get("role", "?")).upper()
            parts: list[str] = []
            if role == "TOOL":
                name = message.get("name")
                if name:
                    parts.append(f"name={name}")
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                names = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        fn = tool_call.get("function") or {}
                        names.append(str(fn.get("name") or tool_call.get("name") or "?"))
                if names:
                    parts.append(f"tool_calls={', '.join(names)}")

            content = message.get("content")
            if isinstance(content, list):
                rendered = [_stringify_block(block) for block in content]
                text = "\n".join(block for block in rendered if block.strip())
            else:
                text = _stringify_block(content)
            text = text.strip()
            if len(text) > cls._MAX_MESSAGE_CHARS:
                text = text[:cls._MAX_MESSAGE_CHARS] + "\n... (truncated)"
            meta = f" [{' | '.join(parts)}]" if parts else ""
            lines.append(f"{idx}. {role}{meta}\n{text or '(empty)'}")
        return "\n\n".join(lines)

    async def review_turn(
        self,
        *,
        turn_messages: list[dict[str, Any]],
        channel: str | None = None,
        chat_id: str | None = None,
        agent_label: str = "main agent",
    ) -> bool:
        """Run the teacher review and persist LEARN.md."""
        if not self.is_enabled() or not turn_messages:
            return False

        teacher_prompt = self._read_teacher_prompt().strip()
        if not teacher_prompt:
            return False

        review_prompt = f"""Review this completed {agent_label} run and decide whether it reveals one reusable lesson worth saving.

Current time: {current_time_str(self.timezone)}
Channel: {channel or "(unknown)"}
Chat ID: {chat_id or "(unknown)"}

## Completed Run Trace
{self._format_messages(turn_messages)}

Return at most one markdown bullet item through the save_learning_item tool.
If there is nothing worth adding, return an empty string for learning_item and set summary to "unchanged"."""

        messages = [
            {"role": "system", "content": teacher_prompt},
            {"role": "user", "content": review_prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_learning_item"}}
            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=_SAVE_LEARNINGS_TOOL,
                model=self.model,
                temperature=0,
                tool_choice=forced,
            )
            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=_SAVE_LEARNINGS_TOOL,
                    model=self.model,
                    temperature=0,
                    tool_choice="auto",
                )

            if not response.tool_calls:
                logger.warning(
                    "Teacher review skipped: save_learning_item tool was not called "
                    "(finish_reason={}, content_preview={})",
                    response.finish_reason,
                    (response.content or "")[:200],
                )
                return False

            payload = _normalize_save_learnings_args(response.tool_calls[0].arguments)
            if payload is None:
                logger.warning("Teacher review skipped: invalid save_learning_item payload")
                return False

            learning_item = payload.get("learning_item")
            summary = payload.get("summary", "")
            if not isinstance(learning_item, str):
                logger.warning("Teacher review skipped: learning_item was not a string")
                return False

            learning_item = learning_item.strip()
            if not learning_item:
                logger.info("Teacher review kept LEARN.md unchanged")
                return True

            if not learning_item.startswith("- "):
                learning_item = f"- {learning_item.lstrip('-').strip()}"
            self._append_learning_item(learning_item)
            logger.info("Teacher review appended to LEARN.md ({})", summary or "changed")
            return True
        except Exception:
            logger.exception("Teacher review failed")
            return False
