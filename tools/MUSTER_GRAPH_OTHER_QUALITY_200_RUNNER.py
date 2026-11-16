#!/usr/bin/env python3
"""Concurrent Colab CPU evaluation entry; immutable inputs and partitioned budgets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

OWNER = "Codex Release"
STAGE = "graph_quality_others_20261116"
MAX_CAP = Decimal("30")
MAX_PARALLEL = 3
PRINT_LOCK = threading.Lock()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".next")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def emit(value):
    with PRINT_LOCK:
        print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), flush=True)


def within(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts,
            "unsafe relative path")
    root = Path(root).resolve()
    result = (root / rel).resolve()
    require(result.is_relative_to(root), "path leaves package")
    return result


def validate_batch(package, manifest):
    require(manifest.get("schema") == "muster.graph_quality.others_batch.v1", "batch schema differs")
    require(manifest.get("owner") == OWNER and manifest.get("stage_id") == STAGE,
            "owner or stage differs")
    require(Decimal(str(manifest.get("total_cap_usd"))) == MAX_CAP, "stage cap differs")
    phases = manifest.get("phases", [])
    require(phases and len({p["id"] for p in phases}) == len(phases), "empty or duplicate phases")
    require(sum((Decimal(str(p["cap_usd"])) for p in phases), Decimal(0)) <= MAX_CAP,
            "child budget allocations exceed parent cap")
    files = manifest.get("files", {})
    require(files, "missing input/source file pins")
    for rel, digest in files.items():
        path = within(package, rel)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "input/source SHA mismatch: " + rel)
    for phase in phases:
        require(phase["id"].replace("_", "").isalnum(), "unsafe phase id")
        require(Decimal(str(phase["cap_usd"])) > 0, "nonpositive child budget")
        require(phase["entrypoint"] in files, "entrypoint is not pinned")
        require(within(package, phase["package_dir"]).is_dir(), "child package is absent")
        require(phase.get("workers", 4) in (1, 2, 3, 4), "child concurrency is not admitted")
        require(phase.get("benchmark_scope") and phase.get("required_secrets") is not None,
                "benchmark scope or secret names missing")
    return phases


def complete_child(folder, expected_contract_sha256=None, expected_package_sha256=None):
    """Exit zero is insufficient; require a bound, complete evaluation receipt."""
    path = Path(folder) / "DONE.json"
    if not path.is_file():
        return False
    try:
        done = json.loads(path.read_text())
        if done.get("complete") is not True:
            return False
        summary = Path(folder) / "SUMMARY.json"
        contract = Path(folder) / "RUN_CONTRACT.json"
        if not summary.is_file() or done.get("summary_sha256") != sha(summary):
            return False
        if expected_package_sha256 is not None:
            return done.get("package_manifest_sha256") == expected_package_sha256
        if not contract.is_file():
            return False
        actual_contract = canonical(json.loads(contract.read_text()))
        return (done.get("summary_sha256") == sha(summary)
                and done.get("contract_sha256") == actual_contract
                and (expected_contract_sha256 is None or actual_contract == expected_contract_sha256))
    except (ValueError, OSError, KeyError):
        return False


def transferable(relative):
    parts = relative.parts
    name = relative.name
    if "__pycache__" in parts or name.endswith((".next", ".lock", "-wal", "-shm")):
        return False
    if relative.suffix in (".sqlite", ".sqlite3", ".db"):
        return "ledger_backup" in parts or name == "ledger_snapshot.sqlite"
    return True


class Mirror:
    """Copy immutable checkpoints and SQLite backup snapshots, never a live WAL DB."""
    def __init__(self, local, remote):
        self.local, self.remote = Path(local), Path(remote)
        self.seen = {}
        self.lock = threading.Lock()

    def restore(self):
        if not self.remote.is_dir():
            return
        for src in sorted(self.remote.rglob("*")):
            if not src.is_file() or src.is_symlink():
                continue
            rel = src.relative_to(self.remote)
            if transferable(rel) and not rel.name.endswith(".upload_tmp"):
                dst = within(self.local, str(rel))
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                require(sha(src) == sha(dst), "checkpoint restore SHA mismatch")

    def sync(self, final=False):
        errors = []
        with self.lock:
            if not self.local.exists():
                return errors
            for src in sorted(self.local.rglob("*")):
                if not src.is_file() or src.is_symlink():
                    continue
                rel = src.relative_to(self.local)
                if not transferable(rel):
                    continue
                try:
                    stamp = (src.stat().st_size, src.stat().st_mtime_ns)
                    if not final and self.seen.get(str(rel)) == stamp:
                        continue
                    before = sha(src)
                    if src.name == "ledger_snapshot.sqlite":
                        metadata = src.parent / "LEDGER_SNAPSHOT.json"
                        if not metadata.exists() or json.loads(metadata.read_text()).get("sha256") != before:
                            if final:
                                errors.append({"file": str(rel), "type": "UnboundLedgerSnapshot"})
                            continue
                    dst = within(self.remote, str(rel))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    temp = dst.with_name(dst.name + ".upload_tmp")
                    shutil.copyfile(src, temp)
                    if sha(src) != before:
                        temp.unlink(missing_ok=True)
                        continue
                    require(sha(temp) == before, "checkpoint write SHA mismatch")
                    temp.replace(dst)
                    self.seen[str(rel)] = stamp
                except (OSError, ValueError) as ex:
                    errors.append({"file": str(rel), "type": type(ex).__name__})
            if final:
                require(not errors, "final Drive checkpoint verification failed")
        return errors


class Children:
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = {}
        self.stopping = threading.Event()

    def stop_owned(self):
        self.stopping.set()
        with self.lock:
            processes = list(self.processes.values())
        for sig, seconds in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
            alive = [p for p in processes if p.poll() is None]
            if not alive:
                return
            for p in alive:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
            until = time.monotonic() + seconds
            while time.monotonic() < until and any(p.poll() is None for p in alive):
                time.sleep(0.1)

    def run(self, phase, package, folder, python, env):
        if complete_child(folder, phase.get("expected_contract_sha256"), phase.get("expected_package_manifest_sha256")):
            emit({"phase": phase["id"], "status": "verified_complete_reuse", "new_processes": 0})
            return {"id": phase["id"], "status": "DONE", "reused": True}
        missing = [name for name in phase["required_secrets"] if not env.get(name)]
        if missing:
            write_json(Path(folder) / "BLOCKED.json", {"reason": "missing_secrets", "names": missing})
            emit({"phase": phase["id"], "status": "BLOCKED", "missing_secret_names": missing})
            return {"id": phase["id"], "status": "BLOCKED", "reason": "missing_secrets"}
        command = [str(python), "-u", str(within(package, phase["entrypoint"])),
                   "--package", str(within(package, phase["package_dir"])),
                   "--out", str(folder), "--mode", "run", "--workers", str(phase.get("workers", 4)),
                   "--cap-usd", str(phase["cap_usd"])]
        emit({"phase": phase["id"], "status": "starting", "scope": phase["benchmark_scope"],
              "cap_usd": str(phase["cap_usd"]), "workers": phase.get("workers", 4)})
        log = Path(folder) / "CHILD_STDOUT.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open("a", encoding="utf-8") as f:
            with self.lock:
                require(not self.stopping.is_set(), "parent is stopping")
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env, start_new_session=True)
                self.processes[phase["id"]] = proc
            assert proc.stdout is not None
            with proc.stdout:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
                    emit("[" + phase["id"] + "] " + line.rstrip())
            code = proc.wait()
        state = "DONE" if code == 0 and complete_child(folder, phase.get("expected_contract_sha256"),
                                                       phase.get("expected_package_manifest_sha256")) else "PARTIAL"
        return {"id": phase["id"], "status": state, "returncode": code,
                "elapsed_seconds": round(time.monotonic() - started, 1), "reused": False}


def execute(package, manifest, local, remote, python, env, *, interval=20):
    phases = validate_batch(package, manifest)
    local, remote = Path(local), Path(remote)
    mirror = Mirror(local, remote)
    mirror.restore()
    expected = {"schema": "muster.graph_quality.others_parent_run.v1", "owner": OWNER,
                "stage_id": STAGE, "batch_sha256": canonical(manifest), "total_cap_usd": str(MAX_CAP),
                "partitions": [{"id": p["id"], "cap_usd": str(p["cap_usd"])} for p in phases]}
    path = local / "PARENT_CONTRACT.json"
    if path.exists():
        require(json.loads(path.read_text()) == expected, "parent output belongs to another batch")
    write_json(path, expected)
    children = Children()
    results, pending = [], {}
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
    try:
        pending = {pool.submit(children.run, p, package, local / p["id"], python, env): p for p in phases}
        while pending:
            done, _ = wait(pending, timeout=interval, return_when=FIRST_COMPLETED)
            for future in done:
                phase = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as ex:
                    results.append({"id": phase["id"], "status": "BLOCKED", "type": type(ex).__name__})
            errors = mirror.sync()
            progress = {"stage": STAGE, "phase_results": sorted(results, key=lambda x: x["id"]),
                        "pending_phases": [p["id"] for p in pending.values()],
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                        "checkpoint_copy_errors": len(errors), "total_cap_usd": str(MAX_CAP)}
            write_json(local / "PARENT_PROGRESS.json", progress)
            emit(progress)
    except BaseException:
        children.stop_owned()
        mirror.sync()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    done = {"status": "DONE" if all(r["status"] == "DONE" for r in results) else "PARTIAL",
            "owner": OWNER, "stage_id": STAGE, "batch_sha256": canonical(manifest),
            "phase_results": sorted(results, key=lambda x: x["id"]),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "all_planned_benchmarks_finished": False,
            "note": "This receipt covers only the fixed phases in this batch; blocked or absent benchmarks are not scored."}
    write_json(local / "PARENT_DONE.json", done)
    mirror.sync(final=True)
    emit(done)
    return done


def run(args):
    package = Path(args.package).resolve()
    manifest_path = package / "BATCH.json"
    manifest = json.loads(manifest_path.read_text())
    require(sha(manifest_path) == args.manifest_sha256, "batch manifest bytes differ")
    phases = validate_batch(package, manifest)
    if args.mode == "plan":
        emit({"status": "PLAN_VERIFIED", "phases": [{k: p[k] for k in
              ("id", "benchmark_scope", "cap_usd", "required_secrets")} for p in phases],
              "total_cap_usd": str(MAX_CAP), "actual_API_calls": 0})
        return
    require(sys.platform == "linux" and Path("/content").is_dir(), "real execution is Colab only")
    import google.colab  # noqa: F401; installed Colab runtime required before children.
    out, local = Path(args.out).resolve(), Path(args.local).resolve()
    require(str(local).startswith("/content/") and "/drive/" not in str(local), "active output must be local")
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    with (local / "parent_execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = out / "ACTIVE_OWNER.json"
        token = uuid.uuid4().hex
        owner = {"owner": OWNER, "stage_id": STAGE, "status": "running", "host": socket.gethostname(),
                 "pid": os.getpid(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                 "token": token, "batch_sha256": canonical(manifest), "heartbeat": time.time()}
        if claim.exists():
            old = json.loads(claim.read_text())
            require(old.get("status") == "released", "existing owner needs reconciliation; no duplicate API calls started")
        write_json(claim, owner)
        time.sleep(2)
        require(json.loads(claim.read_text()).get("token") == token, "another parent acquired this batch")
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(20):
                owner["heartbeat"] = time.time()
                write_json(claim, owner)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        orderly = False
        try:
            result = execute(package, manifest, local / "files", out / "files", args.python, env)
            orderly = True
            return result
        finally:
            stop.set()
            thread.join(timeout=1)
            owner.update(status="released" if orderly else "interrupted_requires_reconciliation", heartbeat=time.time())
            write_json(claim, owner)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--mode", choices=("plan", "run"), default="plan")
    p.add_argument("--out", required=True)
    p.add_argument("--local", required=True)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()
    result = run(args)
    if result is not None and result["status"] != "DONE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
