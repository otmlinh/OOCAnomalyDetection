import os
import time
from datetime import datetime

class Logger:
    def __init__(self, out_dir: str, name: str):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"{name}_{ts}.log")
        self.f = open(self.path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.log(f"[Logger] Writing to {self.path}")

    def log(self, msg: str):
        dt = time.time() - self.t0
        line = f"[{dt:9.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.log("[Logger] Closed.")
        self.f.close()
