import asyncio
import functools
import inspect
import logging
import math
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

def timed(func: Callable) -> Callable:
    name = func.__qualname__

    if inspect.isasyncgenfunction(func):
        @functools.wraps(func)
        async def gen_wrapper(*args: Any, **kwargs: Any):
            started = time.perf_counter()
            try:
                async for item in func(*args, **kwargs):
                    yield item
            finally:
                logger.info("timing function=%s seconds=%.3f", name, time.perf_counter() - started)

        return gen_wrapper

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            logger.info("timing function=%s seconds=%.3f", name, time.perf_counter() - started)

    return wrapper


def is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    if isinstance(exc, httpx.HTTPStatusError):
        value = exc.response.headers.get("retry-after")
        if value:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def retry_with_backoff(
    attempts: int = 3,
    initial_delay: float = 0.5,
    multiplier: float = 2.0,
    max_delay: float = 10.0,
    retry_on: Callable[[BaseException], bool] = is_transient_error,
):
    
    def decorator(func: Callable) -> Callable:
        name = func.__qualname__

        async def _sleep_before_retry(exc: Exception, attempt: int, delay: float) -> float:
            wait = min(max(delay, _retry_after_seconds(exc) or 0.0), max_delay)
            logger.warning(
                "retry function=%s attempt=%s/%s wait=%.1fs error=%r",
                name, attempt, attempts, wait, exc,
            )
            await asyncio.sleep(wait)
            return min(delay * multiplier, max_delay)

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def gen_wrapper(*args: Any, **kwargs: Any):
                delay = initial_delay
                for attempt in range(1, attempts + 1):
                    yielded = False
                    try:
                        async for item in func(*args, **kwargs):
                            yielded = True
                            yield item
                        return
                    except Exception as exc:
                        if yielded or attempt == attempts or not retry_on(exc):
                            raise
                        delay = await _sleep_before_retry(exc, attempt, delay)

            return gen_wrapper

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            delay = initial_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == attempts or not retry_on(exc):
                        raise
                    delay = await _sleep_before_retry(exc, attempt, delay)
        return wrapper
    return decorator

def rate_limit(max_calls: int, period: float):
    def decorator(func: Callable) -> Callable:
        hits: dict[str, deque[float]] = defaultdict(deque)
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            request = kwargs.get("request") or next(
                (a for a in args if isinstance(a, Request)), None
            )
            key = request.client.host if request is not None and request.client else "anonymous"

            now = time.monotonic()
            window = hits[key]
            while window and now - window[0] >= period:
                window.popleft()

            if len(window) >= max_calls:
                retry_after = max(1, math.ceil(period - (now - window[0])))
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_calls} requests per {period:g}s.",
                    headers={"Retry-After": str(retry_after)},
                )

            window.append(now)
            return await func(*args, **kwargs)

        wrapper.reset = hits.clear
        return wrapper

    return decorator
