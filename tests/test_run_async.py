"""Tests for run_async — centralized sync-from-async helper."""
import asyncio
import pytest

from src._async_utils import run_async


async def _echo(value):
    """Simple coroutine that returns its argument."""
    return value


async def _failing():
    """Coroutine that raises."""
    raise ValueError("boom")


class TestRunAsync:
    """run_async() runs coroutines from synchronous code."""

    def test_returns_coroutine_result(self):
        assert run_async(_echo(42)) == 42

    def test_returns_string(self):
        assert run_async(_echo("hello")) == "hello"

    def test_returns_none(self):
        assert run_async(_echo(None)) is None

    def test_propagates_exception(self):
        with pytest.raises(ValueError, match="boom"):
            run_async(_failing())

    def test_works_without_running_loop(self):
        """In a normal script context (no running loop), should work fine."""
        result = run_async(_echo("no_loop"))
        assert result == "no_loop"

    def test_multiple_sequential_calls(self):
        """Can be called multiple times in sequence."""
        results = [run_async(_echo(i)) for i in range(5)]
        assert results == [0, 1, 2, 3, 4]
