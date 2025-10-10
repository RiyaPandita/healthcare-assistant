import csv
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_path="logs/events.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ts","agent","step","detail"])
                writer.writeheader()

    def log(self, agent: str, step: str, detail: dict):
        row = {
            "ts": datetime.utcnow().isoformat(),
            "agent": agent,
            "step": step,
            "detail": str(detail)
        }
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        return row