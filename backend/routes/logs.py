from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json

router = APIRouter()

# In-memory log buffer (most recent 200 events)
_log_buffer: list[dict] = []
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "honeypot_events.jsonl"


class LogEntry(BaseModel):
    eventid: str
    username: Optional[str] = None
    password: Optional[str] = None
    ip: Optional[str] = None
    commands: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@router.post("/")
async def create_log(log: LogEntry):
    entry = log.dict()
    entry["timestamp"] = log.timestamp.isoformat()
    _log_buffer.append(entry)

    # Keep in-memory buffer bounded
    if len(_log_buffer) > 200:
        _log_buffer.pop(0)

    # Persist to JSONL file
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write honeypot log to file: {e}")

    print(f"\n[ESP8266 EVENT] {entry['eventid']} from {entry.get('ip', 'unknown')}")
    if entry.get("username") or entry.get("password"):
        print(f"  Credentials: username='{entry.get('username')}' password='{entry.get('password')}'")
    if entry.get("commands"):
        print(f"  Payload/Cmd: {entry.get('commands')}")

    return {"status": "success", "eventid": log.eventid, "received_at": entry["timestamp"]}


@router.get("/")
async def get_logs(limit: int = 50):
    """Retrieve recent honeypot security events."""
    return _log_buffer[-limit:]


@router.delete("/")
async def clear_logs():
    """Clear in-memory honeypot logs."""
    _log_buffer.clear()
    return {"status": "cleared"}

