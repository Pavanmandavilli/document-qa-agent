import asyncio
import json
import sys

import websockets

async def main(url: str) -> None:
    async with websockets.connect(url) as ws:
        print(f"connected to {url}; ask a question in another terminal...")
        async for message in ws:
            event = json.loads(message)
            extras = {k: v for k, v in event.items() if k not in {"request_id", "agent", "status"}}
            print(f"[{event.get('request_id', '-')}] {event.get('agent'):<12} "
                  f"{event.get('status', ''):<12} {extras or ''}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/progress"))
