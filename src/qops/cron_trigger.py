from __future__ import annotations

from datetime import datetime, timezone

from qops.infra.redis_bus import publish
from qops.market_calendar import todays_session


def main() -> None:
    sess = todays_session()
    if not sess.is_open:
        return

    publish("qops:cmd", {"type": "RUN_NOW", "mode": "paper", "ts": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()