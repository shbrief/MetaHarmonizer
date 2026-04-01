"""Async helpers shared across the project."""
import asyncio


def run_async(coro):
    """Run a coroutine from synchronous code, including Jupyter notebooks.

    If an event loop is already running (e.g. inside a notebook kernel),
    ``nest_asyncio`` is applied so ``run_until_complete`` works without error.
    Otherwise a fresh loop is created and closed after the coroutine finishes.

    Requires ``pip install metaharmonizer[notebook]`` for Jupyter support.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(
                "nest-asyncio is required to run async code inside a "
                "Jupyter notebook. Install it with: "
                "pip install metaharmonizer[notebook]"
            ) from None
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
