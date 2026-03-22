from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException

from qops.infra.redis_bus import client, get_json, publish
from qops.pipeline.run_once import run_once

app = FastAPI()

CMD_CHANNEL = "qops:cmd"

# Redis keys used by API
K_LAST_RUN = "qops:state:last_run"
K_LAST_ERR = "qops:state:last_error"
K_CONSUMER_LOCK = "qops:lock:consumer"

CONSUMER_LOCK_SECONDS = int(os.getenv("CONSUMER_LOCK_SECONDS", "86400"))  # 24h default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_json(key: str, value: Any, ex_seconds: int = 24 * 3600) -> None:
    """Small helper since redis_bus exposes get_json but not necessarily set_json."""
    r = client()
    r.set(key, json.dumps(value), ex=ex_seconds)


def _require_token(x_api_token: Optional[str]) -> None:
    """
    If QOPS_API_TOKEN is set, require it in X-Api-Token header.
    If it's not set, allow (dev-friendly).
    """
    expected = os.getenv("QOPS_API_TOKEN", "").strip()
    if not expected:
        return
    if not x_api_token or x_api_token.strip() != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _decode_pubsub_data(data: Any) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    if isinstance(data, str):
        return data
    # last resort: JSON-serialize
    return json.dumps(data)


def _consumer_loop() -> None:
    r = client()
    pubsub = r.pubsub()
    pubsub.subscribe(CMD_CHANNEL)

    for msg in pubsub.listen():
        if msg.get("type") != "message":
            continue

        try:
            raw = _decode_pubsub_data(msg.get("data"))
            payload = json.loads(raw)

            if payload.get("type") == "RUN_NOW":
                mode = payload.get("mode", "paper")
                run_once(mode=mode)

        except Exception as e:
            # Never crash the consumer thread; report to Redis for /state visibility
            _set_json(
                K_LAST_ERR,
                {
                    "ts": _utc_now(),
                    "where": "consumer_loop",
                    "error": str(e),
                    "msg_type": msg.get("type"),
                },
                ex_seconds=24 * 3600,
            )


def _try_start_consumer() -> None:
    """
    Prevent duplicate consumer threads if multiple uvicorn workers/reload
    ever happen. Uses a Redis NX lock.
    """
    r = client()
    acquired = r.set(K_CONSUMER_LOCK, "1", nx=True, ex=CONSUMER_LOCK_SECONDS)
    if not acquired:
        # Another instance/worker already started the consumer.
        return

    t = threading.Thread(target=_consumer_loop, daemon=True, name="qops-consumer")
    t.start()


@app.on_event("startup")
def startup() -> None:
    _try_start_consumer()


@app.get("/")
def home() -> Dict[str, Any]:
    return {"ok": True, "service": "qops-api"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "qops-api",
        "ts": _utc_now(),
    }


@app.post("/run")
def run_now(
    mode: str = "paper",
    x_api_token: Optional[str] = Header(default=None, alias="X-Api-Token"),
) -> Dict[str, Any]:
    _require_token(x_api_token)

    publish(
        CMD_CHANNEL,
        {"type": "RUN_NOW", "mode": mode, "ts": _utc_now()},
    )
    return {"queued": True, "cmd": "RUN_NOW", "mode": mode}


@app.get("/state")
def state() -> Dict[str, Any]:
    # Normalize missing keys to {}
    return {
        "last_run": get_json(K_LAST_RUN) or {},
        "last_error": get_json(K_LAST_ERR) or {},
        "last_candidates": get_json("qops:state:last_candidates") or {},
        "last_gate": get_json("qops:state:last_gate") or {},
        "last_orders": get_json("qops:state:last_orders") or {},
    }