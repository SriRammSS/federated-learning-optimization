
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

from .io import write_csv


class ResourceWatchdog:
    def __init__(self,out_dir:Path,interval_seconds:float=30.0,warn_gb:float=34.0,
                 pause_gb:float=38.0,stop_gb:float=42.0):
        self.out_dir=out_dir
        self.interval_seconds=interval_seconds
        self.warn_gb=warn_gb
        self.pause_gb=pause_gb
        self.stop_gb=stop_gb
        self.rows:list[dict]=[]
        self.events:list[dict]=[]
        self._stop=threading.Event()
        self._thread:threading.Thread | None=None
        self._stage="idle"

    def start(self) -> None:
        self.out_dir.mkdir(parents=True,exist_ok=True)
        self._thread=threading.Thread(target=self._loop,daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.flush()

    def set_stage(self,stage:str) -> None:
        self._stage=stage

    def check(self,stage:str | None=None) -> str:
        row=sample_resources(stage or self._stage)
        self.rows.append(row)
        status="ok"
        if row["memory_used_gb"] >= self.stop_gb:
            status="stop"
        elif row["memory_used_gb"] >= self.pause_gb:
            status="pause_optional"
        elif row["memory_used_gb"] >= self.warn_gb:
            status="warn"
        if status != "ok":
            event={**row,"event":status}
            self.events.append(event)
            write_csv(self.out_dir / "oom_guard_events.csv",self.events)
        return status

    def should_skip_optional(self) -> bool:
        return self.check() in {"pause_optional","stop"}

    def flush(self) -> None:
        write_csv(self.out_dir / "resource_timeseries.csv",self.rows)
        write_csv(self.out_dir / "oom_guard_events.csv",self.events)
        if self.rows:
            latest=self.rows[-1]
            summary=[{
                "samples":len(self.rows),
                "memory_used_gb_max":max(float(r["memory_used_gb"]) for r in self.rows),
                "memory_free_gb_min":min(float(r["memory_free_gb"]) for r in self.rows),
                "swap_used_gb_max":max(float(r["swap_used_gb"]) for r in self.rows),
                "latest_stage":latest["stage"],
                "platform":platform.platform(),
            }]
            write_csv(self.out_dir / "resource_summary.csv",summary)

    def _loop(self):
        while not self._stop.is_set():
            self.check(self._stage)
            self.flush()
            self._stop.wait(self.interval_seconds)


def sample_resources(stage:str) -> dict:
    mem=_memory_gb()
    return {"timestamp":time.time(),"stage":stage,"pid":os.getpid(),**mem}


def _memory_gb():
    if platform.system() == 'Darwin':
        return _darwin_memory_gb()
    return _linux_memory_gb()


def _linux_memory_gb() -> dict:
    vals={}
    try:
        text=Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            key,raw=line.split(":",1)
            vals[key]=float(raw.strip().split()[0]) / (1024 * 1024)
        total=vals.get("MemTotal",0.0)
        free=vals.get("MemAvailable",vals.get("MemFree",0.0))
        swap_total=vals.get("SwapTotal",0.0)
        swap_free=vals.get("SwapFree",0.0)
        return {
            "memory_total_gb":total,
            "memory_free_gb":free,
            "memory_used_gb":max(0.0,total - free),
            "swap_used_gb":max(0.0,swap_total - swap_free),
            "process_rss_gb":_process_rss_gb(),
        }
    except Exception:
        return _fallback_memory_gb()


def _darwin_memory_gb():
    try:
        page_size=16384.0
        total=int(subprocess.check_output(["/usr/sbin/sysctl","-n","hw.memsize"],text=True).strip()) / 1e9
        vm=subprocess.check_output(["/usr/bin/vm_stat"],text=True)
        vals={}
        for line in vm.splitlines():
            if ":" not in line:
                continue
            key,raw=line.split(":",1)
            if not key.strip().startswith("Pages"):
                continue
            vals[key.strip()]=_parse_vm_pages(raw)
        free_pages=vals.get("Pages free",0.0) + vals.get("Pages speculative",0.0)
        compressor_pages=vals.get("Pages occupied by compressor",vals.get("Pages used by compressor",0.0))
        free=free_pages * page_size / 1e9
        compressor=compressor_pages * page_size / 1e9
        return {
            "memory_total_gb":total,
            "memory_free_gb":_darwin_pressure_free_gb(total,free),
            "memory_used_gb":max(0.0,total - _darwin_pressure_free_gb(total,free)),
            "swap_used_gb":_darwin_swap_used_gb(),
            "compressor_gb":compressor,
            "process_rss_gb":_process_rss_gb(),
        }
    except Exception:
        return _fallback_memory_gb()


def _darwin_swap_used_gb():
    try:
        text=subprocess.check_output(["/usr/sbin/sysctl","-n","vm.swapusage"],text=True)
        marker="used = "
        if marker in text:
            raw=text.split(marker,1)[1].split()[0]
            return float(raw.rstrip("M")) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def _darwin_pressure_free_gb(total_gb:float,fallback_free_gb:float) -> float:
    try:
        text=subprocess.check_output(["/usr/bin/memory_pressure"],text=True)
        marker="System-wide memory free percentage:"
        for line in text.splitlines():
            if marker in line:
                pct=float(line.split(marker,1)[1].strip().rstrip("%"))
                return total_gb * pct / 100.0
    except Exception:
        pass
    return fallback_free_gb


def _parse_vm_pages(raw:str) -> float:
    token=raw.strip().strip(".").split()[0]
    return float(token.replace(".",""))


def _process_rss_gb():
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return rss / 1e9
        return rss / (1024 * 1024)
    except Exception:
        return 0.0


def _fallback_memory_gb():
    return {
        "memory_total_gb":0.0,
        "memory_free_gb":0.0,
        "memory_used_gb":0.0,
        "swap_used_gb":0.0,
        "process_rss_gb":_process_rss_gb(),
    }
