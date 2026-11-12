from __future__ import annotations

import json
import time
from pathlib import Path


class CooldownStore:
    def __init__(self, path: str | Path, cooldown_seconds: int = 300):
        self.path = Path(path)
        self.cooldown_seconds = cooldown_seconds

    def _read(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def may_send(self, event_id: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        previous = self._read().get(event_id)
        return previous is None or now - previous >= self.cooldown_seconds

    def record(self, event_id: str, now: float | None = None) -> None:
        data = self._read()
        data[event_id] = time.time() if now is None else now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(self.path)
