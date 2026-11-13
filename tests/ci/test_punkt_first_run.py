"""First-use asset registration and failed-download reporting, without network calls."""
import importlib
from types import SimpleNamespace

import nltk
import pytest

from muster.errors import AssetMissing


cut_module = importlib.import_module("muster.cut")


def isolated_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(cut_module.bpe, "asset_dir", lambda: str(tmp_path / "new-assets"))
    monkeypatch.setattr(nltk.data, "path", [])
    return str(tmp_path / "new-assets" / "nltk_data")


def test_download_registers_its_path_before_nltk_security_checks(tmp_path, monkeypatch):
    destination = isolated_assets(tmp_path, monkeypatch)
    events = []
    installed = []

    def find(resource):
        events.append("find")
        assert resource == "tokenizers/punkt_tab"
        if not installed:
            raise LookupError("first use has no sentence splitter")
        return destination

    def download(resource, *, download_dir, quiet):
        events.append("download")
        assert resource == "punkt_tab" and download_dir == destination and quiet is True
        assert destination in nltk.data.path
        installed.append(True)
        return True

    monkeypatch.setattr(nltk.data, "find", find)
    monkeypatch.setattr(nltk, "download", download)
    assert cut_module._ensure_punkt(exact=False) is nltk
    assert events == ["find", "download", "find"]


def test_failed_download_cannot_print_a_false_ready_message(tmp_path, monkeypatch, capsys):
    from muster.cli import cmd_fetch
    destination = isolated_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(cut_module.bpe, "fetch", lambda **kwargs: [])

    def missing(resource):
        raise LookupError("sentence splitter still missing")

    def failed(*args, **kwargs):
        assert destination in nltk.data.path
        return False

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", failed)
    with pytest.raises(LookupError, match="still missing"):
        cmd_fetch(SimpleNamespace(models=False))
    assert "ready" not in capsys.readouterr().out


def test_exact_missing_asset_refusal_never_downloads(tmp_path, monkeypatch):
    isolated_assets(tmp_path, monkeypatch)

    def missing(resource):
        raise LookupError("absent")

    def forbidden(*args, **kwargs):
        pytest.fail("exact replay attempted a download")

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", forbidden)
    with pytest.raises(AssetMissing, match="muster fetch"):
        cut_module._ensure_punkt(exact=True)
