"""Tests for the ThermDash local web UI HTTP server and API endpoints."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from token_dashboard.web_ui import create_server


@pytest.fixture()
def ui_server_url():
    server = create_server(host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _post_json(url: str, payload: dict) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, dict(resp.headers), resp.read()


def test_ui_index_and_models(ui_server_url: str):
    with urllib.request.urlopen(f"{ui_server_url}/", timeout=5) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "ThermDash Complete" in html

    with urllib.request.urlopen(f"{ui_server_url}/api/models", timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        model_names = {m["model"] for m in data["models"]}
        assert "o3-mini" in model_names
        assert "gpt-4o" in model_names


def test_ui_calc_energy_and_error(ui_server_url: str):
    status, _, raw = _post_json(
        f"{ui_server_url}/api/calc-energy",
        {"model": "o3-mini", "tokens_in": 1200, "tokens_out": 500, "rework_turns": 2},
    )
    assert status == 200
    data = json.loads(raw.decode("utf-8"))
    assert data["model"] == "o3-mini"
    assert data["e_in_joules"] > 0
    assert data["rework_turns"] == 2

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{ui_server_url}/api/calc-energy", {"model": "nonexistent-model-xyz", "tokens_in": 100, "tokens_out": 50})
    assert exc_info.value.code == 400


def test_ui_audit_n8n_and_pdf(ui_server_url: str):
    sample_wf = {
        "name": "UI Test Workflow",
        "nodes": [
            {"name": "OpenAI Chat", "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi", "parameters": {"model": "gpt-4o"}}
        ],
        "connections": {},
    }
    status, _, raw = _post_json(f"{ui_server_url}/api/audit-n8n", {"workflow": sample_wf})
    assert status == 200
    report = json.loads(raw.decode("utf-8"))
    assert report["workflow_name"] == "UI Test Workflow"
    assert report["estimated_run_joules"] > 0
    assert "OpenAI Chat" in report["inventory_csv"]

    status, headers, pdf_bytes = _post_json(
        f"{ui_server_url}/api/audit-n8n/pdf",
        {"workflow": sample_wf, "monthly_runs": 500, "hourly_rate": 50},
    )
    assert status == 200
    assert headers.get("Content-Type") == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF")
