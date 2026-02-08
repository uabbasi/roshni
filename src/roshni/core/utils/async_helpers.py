"""Async utilities for running coroutines from synchronous contexts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


def run_async_safely(coro):
    """
    Run an async coroutine from a sync context.

    If no event loop is running, uses asyncio.run() directly.
    If one is already running (e.g. inside FastAPI/Jupyter), dispatches
    to a thread pool to avoid "cannot run nested event loop" errors.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
