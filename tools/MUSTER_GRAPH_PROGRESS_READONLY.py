"""Read-only native FAST progress; no SDK, model, package install, or writes."""
import argparse as _lp_argparse
from collections import deque as _lp_deque
from datetime import datetime as _lp_datetime, timezone as _lp_timezone
import hashlib as _lp_hashlib
import json as _lp_json
from pathlib import Path as _lp_Path
import threading as _lp_threading
import time as _lp_time


def _lp_read(path):
    try:
        raw = path.read_bytes()
        value = _lp_json.loads(raw)
        return (value, path.stat().st_mtime, raw) if isinstance(value, dict) else ({}, None, None)
    except (OSError, ValueError):
        return {}, None, None


def _lp_duration(seconds):
    if seconds is None:
        return "미정"
    minutes, secs = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class LiveProgress:
    """Print actual committed counts; ETA starts after a measured count delta."""

    def __init__(self, root, *, cohort="lmev2_small451", total=451, caps=(200, 400),
                 mirrored=False, interval=30, emit=print, clock=None):
        self.root = _lp_Path(root)
        self.cohort, self.total, self.caps = cohort, total, tuple(caps)
        self.mirrored, self.interval, self.emit = mirrored, interval, emit
        self.clock = clock or _lp_time.time
        self.started = self.clock()
        self.samples = {cap: _lp_deque() for cap in self.caps}
        self.stop = _lp_threading.Event()
        self.thread = None

    def _job(self, cap):
        phase = self.root / f"lmev2_cap{cap}"
        return phase / ("files" if self.mirrored else "") / f"cap{cap}" / self.cohort

    def sample(self):
        now = self.clock()
        rows = []
        for cap in self.caps:
            job = self._job(cap)
            progress, updated, _ = _lp_read(job / "PROGRESS.json")
            done, _, done_raw = _lp_read(job / "DONE.json")
            fast, _, _ = _lp_read(job / "FAST_JOB_DONE.json")
            failed, _, _ = _lp_read(job / "FAILED.json")
            phase = self.root / f"lmev2_cap{cap}"
            blocked, _, _ = _lp_read((phase / "files" if self.mirrored else phase) / "NATIVE_FAST_BLOCKED.json")
            status, _, _ = _lp_read(self.root / f"lmev2_cap{cap}.STATUS.json")
            count = progress.get("completed_queries")
            valid_count = type(count) is int and 0 <= count <= self.total
            if not valid_count:
                count = None
            complete = (fast.get("status") == "DONE" and done.get("complete") is True
                        and done.get("questions") == self.total and done_raw is not None
                        and fast.get("done_sha256") == _lp_hashlib.sha256(done_raw).hexdigest())
            if complete:
                count = self.total
            samples = self.samples[cap]
            reset = count is not None and bool(samples) and count < samples[-1][1]
            if reset:
                samples.clear()
            if count is not None:
                samples.append((now, count))
                while len(samples) > 2 and samples[1][0] < now - 300:
                    samples.popleft()
            rate = eta = None
            if len(samples) > 1:
                dt, dc = now - samples[0][0], count - samples[0][1] if count is not None else 0
                if dt >= 1 and dc > 0:
                    rate = 60 * dc / dt
                    eta = (self.total - count) / (rate / 60)
            if complete:
                state, eta = "완료기록 검증", 0
            elif failed or blocked or status.get("status") in ("BLOCKED", "INTERRUPTED", "PARTIAL"):
                state, eta, rate = "중단/실패 기록", None, None
            elif count == self.total:
                state, eta = "마지막 저장·완료 검증 중", None
            elif count is not None:
                state = "질문 기록 관측"
            elif (job / "RUNTIME_HOOK.json").is_file():
                state = "저장소 준비·첫 질문 대기"
            elif (phase / ("files" if self.mirrored else "") / "NATIVE_FAST_CONFIG.json").is_file():
                state = "검증·모델/저장소 준비"
            else:
                state = "시작 기록 대기"
            age = max(0, now - updated) if updated is not None else None
            if age is not None and age >= 180 and not complete:
                state += " / 3분 이상 새 완료 기록 없음"
            if reset:
                state += " / 카운트 후퇴: ETA 재측정"
            rows.append({"cap": cap, "completed": count, "total": self.total, "state": state,
                         "complete": complete, "rate_q_per_min": rate, "eta_seconds": eta,
                         "record_updated_epoch": updated, "record_age_seconds": age})
        return {"utc": _lp_datetime.fromtimestamp(now, _lp_timezone.utc).strftime("%H:%M:%S UTC"),
                "cohort": self.cohort, "watch_elapsed_seconds": now - self.started,
                "questions_completed": sum(r["completed"] or 0 for r in rows),
                "question_occurrences_total": self.total * len(self.caps), "caps": rows,
                "all_complete": all(r["complete"] for r in rows),
                "read_only": True, "model_API_calls": 0}

    def report(self):
        value = self.sample()
        lines = [f"[{value['utc']}] {self.cohort} | 전체 {value['questions_completed']}/"
                 f"{value['question_occurrences_total']}문맥 | 모니터 경과 {_lp_duration(value['watch_elapsed_seconds'])}"]
        for row in value["caps"]:
            count = "미관측" if row["completed"] is None else str(row["completed"])
            speed = "미정" if row["rate_q_per_min"] is None else f"{row['rate_q_per_min']:.2f}문항/분"
            updated = "미관측" if row["record_updated_epoch"] is None else (
                _lp_datetime.fromtimestamp(row["record_updated_epoch"], _lp_timezone.utc).strftime("%H:%M:%S UTC")
                + f" ({int(row['record_age_seconds'])}초 전)")
            lines.append(f"  그래프 {row['cap']} | {count}/{row['total']} | {row['state']} | "
                         f"최근 속도 {speed} | ETA {_lp_duration(row['eta_seconds'])} | 기록 갱신 {updated}")
        lines.append("  완료 수는 저장된 질문 기록입니다. GPU 생존·정답률을 뜻하지 않습니다. ETA는 관측 후 남은 해당 상한의 추정입니다.")
        self.emit("\n".join(lines), flush=True)
        return value

    def watch(self):
        while not self.stop.is_set():
            if self.report()["all_complete"]:
                return
            self.stop.wait(self.interval)

    def __enter__(self):
        self.report()
        def loop():
            while not self.stop.wait(self.interval):
                if self.report()["all_complete"]:
                    break
        self.thread = _lp_threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.report()


def progress_main():
    parser = _lp_argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/content/muster_graph_fast_work/cea67a239526543e")
    parser.add_argument("--drive", action="store_true", help="read the existing Drive checkpoints")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    watcher = LiveProgress(args.root, mirrored=args.drive, interval=args.interval)
    watcher.report() if args.once else watcher.watch()


if __name__ == "__main__":
    progress_main()
