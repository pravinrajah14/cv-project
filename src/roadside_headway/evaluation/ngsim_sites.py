from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")

# Collection date per NGSIM site, from each site's metadata documentation.
SITE_DATES = {
    "us-101": "2005-06-15",
    "lankershim": "2005-06-16",
    "i-80": "2005-04-13",
    "peachtree": "2006-11-08",
}

TIME_WINDOWS = ["0750am-0805am", "0805am-0820am", "0820am-0835am"]


def _parse_local(location: str, time_str: str) -> datetime.datetime:
    date_str = SITE_DATES[location]
    return datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %I%M%p").replace(tzinfo=PACIFIC)


def nominal_epoch_range_ms(location: str, time_window: str, buffer_s: int = 90) -> tuple[int, int]:
    """Nominal [start, end] epoch-ms range for a site's 15-minute time window,
    padded by `buffer_s` on each side (recording sometimes starts slightly
    before/after the nominal wall-clock label — confirmed for the default
    us-101/0750am-0805am window, where the earliest Global_Time in the ground
    truth is ~21s before 07:50:00 local)."""
    start_str, end_str = time_window.split("-")
    start = _parse_local(location, start_str)
    end = _parse_local(location, end_str)
    return (
        int(start.timestamp() * 1000) - buffer_s * 1000,
        int(end.timestamp() * 1000) + buffer_s * 1000,
    )
