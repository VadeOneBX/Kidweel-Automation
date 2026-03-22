from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest


NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    is_open: bool
    open_dt: datetime | None
    close_dt: datetime | None


def _trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    return TradingClient(key, secret, paper=paper)


def todays_session() -> Session:
    tc = _trading_client()
    d0 = date.today()

    cal = tc.get_calendar(GetCalendarRequest(start=d0, end=d0))
    if not cal:
        return Session(False, None, None)

    # Alpaca calendar entries have date + open/close (times in ET)
    c = cal[0]
    open_dt = datetime.combine(c.date, c.open, tzinfo=NY)
    close_dt = datetime.combine(c.date, c.close, tzinfo=NY)
    return Session(True, open_dt, close_dt)