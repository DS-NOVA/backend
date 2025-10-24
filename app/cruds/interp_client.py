# app/clients/interp_client.py

import httpx
import asyncio
from typing import Optional

class InterpClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0, max_concurrency: int = 8, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.retries = retries
        self.sem = asyncio.Semaphore(max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def interpolate_pair(self, img0_bytes: bytes, img1_bytes: bytes) -> bytes:
        await self.start()
        for _ in range(self.retries + 1):
            try:
                async with self.sem:
                    files = {
                        "img0": ("img0.png", img0_bytes, "image/png"),
                        "img1": ("img1.png", img1_bytes, "image/png"),
                    }
                    resp = await self._client.post("/interpolate", files=files)
                resp.raise_for_status()
                return resp.content  # PNG bytes
            except Exception as e:
                last_exc = e
        raise last_exc
