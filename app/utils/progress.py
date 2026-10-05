import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ProgressBus:
    def __init__(self, max_queue: int = 100):
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._max_queue = max_queue

    def subscribe(self) -> "asyncio.Queue[dict[str, Any]]":
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[dict[str, Any]]") -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("progress subscriber too slow; dropping event")

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
