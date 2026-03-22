from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

def client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)

def publish(channel: str, payload: Dict[str, Any]) -> None:
    r = client()
    r.publish(channel, json.dumps(payload))

def set_json(key: str, payload: Dict[str, Any], ex: Optional[int] = None) -> None:
    r = client()
    r.set(key, json.dumps(payload), ex=ex)

def get_json(key: str) -> Optional[Dict[str, Any]]:
    r = client()
    raw = r.get(key)
    return json.loads(raw) if raw else None