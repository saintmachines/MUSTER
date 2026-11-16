"""MUSTER graph 200/400 fast switch. Generated code-only Colab entrypoint.

Use this in the old MUSTER Colab terminal with --stop-old, or run the generated
notebook. It reads the existing private Drive package and writes a new result
folder. It never deletes baseline files or runs an unbounded new search.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


CODE_PAYLOAD = "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"
CODE_SHA = "cea67a239526543ee07a43bcb4fab8369b2f7bf6021b86d8d08ae91ef7586c3c"
PACKAGE_SHA = "718c6081b56eb47137e87a789791c004036b37bc0de3fba25667e756b3389cc6"
DRIVE_ROOT = Path("/content/drive/MyDrive")
PACKAGE_ROOT = Path("/content/muster_static_source_phase/718c6081b56e/package")
OLD_OUTPUT = DRIVE_ROOT / "MUSTER_native5_full3_static_20261115_v2/outputs/native-static-v1-718c6081b56e"


def drive_mounted():
    # A leftover directory after unmount is not persistent Drive storage.
    return Path("/proc/mounts").is_file() and any(
        len(row.split()) > 1 and row.split()[1] == "/content/drive"
        for row in Path("/proc/mounts").read_text().splitlines())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def unpack_code():
    raw = base64.b64decode(CODE_PAYLOAD, validate=True)
    if hashlib.sha256(raw).hexdigest() != CODE_SHA:
        raise RuntimeError("embedded code ZIP SHA mismatch")
    root = Path("/content/muster_graph_fast_code") / CODE_SHA[:16]
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("unsafe embedded code path")
        manifest = json.loads(z.read("CODE_MANIFEST.json"))
        if set(z.namelist()) != set(manifest["files"]) | {"CODE_MANIFEST.json"}:
            raise RuntimeError("code ZIP inventory differs")
        for name, expected in manifest["files"].items():
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError("embedded source hash mismatch: " + name)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return root, manifest


def prepare_package():
    source = DRIVE_ROOT / "MUSTER_native5_full3_static_20261115_v2/package"
    local_manifest = PACKAGE_ROOT / "PACKAGE_MANIFEST.json"
    src_manifest = source / "PACKAGE_MANIFEST.json"
    mp = local_manifest if local_manifest.is_file() and sha(local_manifest) == PACKAGE_SHA else src_manifest
    if not mp.is_file() or sha(mp) != PACKAGE_SHA:
        raise RuntimeError("existing native package manifest absent or changed")
    info = json.loads(mp.read_text())
    if len(info["files"]) != 60:
        raise RuntimeError("native package file inventory differs")
    PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
    for name, row in info["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("unsafe native package path")
        dst = PACKAGE_ROOT / name
        if dst.is_file() and sha(dst) == row["sha256"]:
            continue
        src = source / name
        if not src.is_file() or sha(src) != row["sha256"]:
            raise RuntimeError("private Drive package file differs: " + name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".copying")
        shutil.copyfile(src, tmp)
        if sha(tmp) != row["sha256"]:
            raise RuntimeError("local package copy differs: " + name)
        os.replace(tmp, dst)
    if mp != local_manifest:
        shutil.copyfile(mp, local_manifest)
    print("기존 입력·휠·생산자 60파일 검증 완료", flush=True)
    return info


def prepare_models(package, cache, output):
    # Preserve the Colab-provided torch build. These are the original phase pins.
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True)
    name, memory = gpu.stdout.strip().splitlines()[0].rsplit(",", 1)
    if "RTXPRO6000" not in name.upper().replace(" ", "") or int(memory) < 90000:
        raise RuntimeError("the frozen native profile requires RTX PRO 6000 with at least 90000 MiB")
    before = importlib.metadata.version("torch")
    pins = {"sentence-transformers": "5.3.0", "transformers": "5.5.0", "rank-bm25": "0.2.2",
            "nltk": "3.10.0", "tiktoken": "0.14.0"}
    missing = []
    for name, version in pins.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            missing.append(name + "==" + version)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall",
                    str(PACKAGE_ROOT / "wheel/muster_memory-1.0.0-py3-none-any.whl")], check=True)
    if importlib.metadata.version("torch") != before:
        raise RuntimeError("Colab torch changed during dependency preparation")
    assets = Path("/content/muster_assembly_parallel_assets")
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TIKTOKEN_CACHE_DIR"] = str(assets / "tiktoken_cache")
    os.environ["NLTK_DATA"] = str(assets / "nltk_data")
    import nltk
    data = Path(os.environ["NLTK_DATA"])
    data.mkdir(parents=True, exist_ok=True)
    if str(data) not in nltk.data.path:
        nltk.data.path.insert(0, str(data))
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", download_dir=str(data), quiet=True):
            raise RuntimeError("NLTK punkt_tab preparation failed")
    import tiktoken
    tiktoken.get_encoding("cl100k_base")
    tiktoken.get_encoding("o200k_base")
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", "-B", str(PACKAGE_ROOT / "code/prepare_pinned_model_cache.py"),
        "--manifest", str(PACKAGE_ROOT / "code/MODEL_FILES.json"),
        "--manifest-sha256", package["model_files_sha256"], "--cache", str(cache),
        "--receipt", str(output / "MODEL_FILES_READBACK.json")],
        check=True, env=dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0"))
    print("고정 모델 준비 완료. 원 torch·FP32·batch32/16·max2048 유지", flush=True)


def run(*, stop_old=False):
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("actual execution is Colab-only")
    if not drive_mounted() or not DRIVE_ROOT.is_dir():
        raise RuntimeError("Google Drive must already be mounted at /content/drive")
    code, metadata = unpack_code()
    stopper = module(code / "stop_old_native.py", "_muster_stop_unbounded")
    if stop_old:
        if stopper.main(["--stop"]):
            raise RuntimeError("old native processes remain; new GPU work has not started")
        stopped = json.loads(Path("/content/MUSTER_NATIVE_STOP_RECEIPT.json").read_text())
        if stopped.get("master_interrupted") is not None:
            # The old master unmounts Drive in its finally block. Never race it.
            raise RuntimeError("old master interrupted and will flush/unmount Drive; stop succeeded. "
                               "Run the new notebook after the old cell stops; it mounts Drive again. "
                               "New GPU work has not started.")
    else:
        if stopper.snapshot():
            raise RuntimeError("old unbounded native work is active on this VM; use --stop-only, "
                               "wait for the old cell to stop, then run the new notebook")
        print("이 VM에 기존 긴 native 작업이 없습니다.", flush=True)
    package = prepare_package()
    if not drive_mounted():
        raise RuntimeError("Drive was unmounted; no new output has been written")
    output = DRIVE_ROOT / "MUSTER_graph_budget_fast_20261116/results" / CODE_SHA[:16]
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("/content/muster_assembly_parallel_assets/hf")
    prepare_models(package, cache, output)
    runner = module(code / "campaign.py", "_muster_graph_campaign")
    native_rel = "native_fast/native_fast.py"
    shared = {"package_dir": str(PACKAGE_ROOT), "output_dir": "{output}", "models_cache": "{cache}",
              "cohorts": ["lmev2_small451"], "patch_path": "{asset}/native_fast/metadata/search_graphcap.py"}
    # Finish all 451 questions at 200 before beginning 400. No unbounded job.
    phases = []
    for cap in (200, 400):
        phase_config = dict(shared, caps=[cap], baseline_dir=str(OLD_OUTPUT))
        if cap == 400:
            phase_config["baseline_dir"] = "{local}/lmev2_cap200/cap200"
        phases.append({"id": "lmev2_cap" + str(cap), "module": native_rel,
            "module_sha256": metadata["files"][native_rel], "config": phase_config})
    config = {"schema": runner.SCHEMA, "asset_dir": str(code), "output_dir": str(output),
        "local_dir": str(Path("/content/muster_graph_fast_work") / CODE_SHA[:16]),
        "models_cache": str(cache), "phases": phases}
    print("실행: LongMemEval v2 451문항 × 그래프200 → 그래프400. 답/judge API 0", flush=True)
    print("새 결과 폴더:", output, flush=True)
    result = runner.run_campaign(config)
    print("전환 실행 상태:", result["status"], flush=True)
    print("검색·조립 기록입니다. 답 정확도는 별도로 측정해야 합니다.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-old", action="store_true")
    parser.add_argument("--stop-only", action="store_true", help="Stop the old VM's native job; do not start new work")
    args = parser.parse_args()
    if args.stop_only:
        code, _ = unpack_code()
        return module(code / "stop_old_native.py", "_muster_stop_unbounded").main(["--stop"])
    return 0 if run(stop_old=args.stop_old)["status"] == "DONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
