"""Lightweight local web UI server for ThermDash Complete (stdlib http.server)."""

from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from typing import Any
import webbrowser

from .energy_table import all_constants, lookup
from .inventory import export_inventory_csv, export_inventory_json, inventory_from_n8n
from .n8n_audit import N8nWorkflowAuditor
from .pdf_report import generate_audit_pdf
from .thermodynamics import MissingEnergyConstantsError, efficiency

_STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


def _calc_energy_payload(body: dict[str, Any]) -> dict[str, Any]:
    model = str(body.get("model", "")).strip()
    c = lookup(model)
    q_raw = body.get("quality")
    q = float(q_raw) if q_raw not in (None, "", "null") else None
    lat_raw = body.get("latency_ms")
    latency_ms = float(lat_raw) if lat_raw not in (None, "", "null") else None

    rep = efficiency(
        n_in=int(body.get("tokens_in", 0)),
        n_out=int(body.get("tokens_out", 0)),
        c=c,
        q=q,
        n_reasoning=int(body.get("reasoning_tokens", 0) or 0),
        rework_turns=max(1, int(body.get("rework_turns", 1) or 1)),
        latency_ms=latency_ms,
    )
    return {
        "model": c.model,
        "gpu": c.gpu,
        "quantization": c.quantization,
        "measured": c.measured,
        "is_reasoning": c.is_reasoning,
        "reasoning_multiplier": c.reasoning_multiplier,
        "source": c.source,
        "e_in_joules": round(rep.e_in, 6),
        "e_in_iqr_low_joules": round(rep.e_in_low, 6),
        "e_in_iqr_high_joules": round(rep.e_in_high, 6),
        "w_useful_joules": round(rep.w_useful, 6),
        "eta": round(rep.eta, 4),
        "waste_share": round(rep.waste_share, 4),
        "rework_turns": rep.rework_turns,
        "latency_ms": rep.latency_ms,
    }


def _audit_n8n_report(body: dict[str, Any]):
    workflow_data = body.get("workflow")
    if isinstance(workflow_data, str):
        workflow_data = json.loads(workflow_data)
    if not isinstance(workflow_data, dict):
        raise ValueError("Missing or invalid 'workflow' JSON object.")

    exec_data = body.get("execution_log")
    if isinstance(exec_data, str) and exec_data.strip():
        exec_data = json.loads(exec_data)
    elif not isinstance(exec_data, dict):
        exec_data = None

    auditor = N8nWorkflowAuditor(workflow_data)
    report = auditor.audit(
        assumed_in_tokens=int(body.get("tokens_in", 1000) or 1000),
        assumed_out_tokens=int(body.get("tokens_out", 400) or 400),
        execution_data=exec_data,
    )
    return report


class ThermDashRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the ThermDash UI and JSON/PDF APIs."""

    server_version = "ThermDashUI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy per-request stderr logs unless debugging
        return

    def _send_bytes(self, status: int, content_type: str, data: bytes, extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not _STATIC_INDEX.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "UI index.html not found"})
                return
            self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", _STATIC_INDEX.read_bytes())
            return

        if path == "/api/models":
            models = [asdict(c) for c in all_constants()]
            self._send_json(HTTPStatus.OK, {"models": models})
            return

        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_json_body()

            if path == "/api/calc-energy":
                self._send_json(HTTPStatus.OK, _calc_energy_payload(body))
                return

            if path == "/api/audit-n8n":
                report = _audit_n8n_report(body)
                items = inventory_from_n8n(report)
                payload = asdict(report)
                payload["inventory"] = [asdict(i) for i in items]
                payload["inventory_csv"] = export_inventory_csv(items)
                payload["inventory_json"] = export_inventory_json(items)
                self._send_json(HTTPStatus.OK, payload)
                return

            if path == "/api/audit-n8n/pdf":
                report = _audit_n8n_report(body)
                monthly_runs = int(body.get("monthly_runs", 1000) or 1000)
                hourly_rate = float(body.get("hourly_rate", 45.0) or 45.0)
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_path = Path(tmpdir) / "thermdash_audit_report.pdf"
                    generate_audit_pdf(
                        report,
                        pdf_path,
                        monthly_runs=monthly_runs,
                        staff_hourly_rate_usd=hourly_rate,
                    )
                    pdf_bytes = pdf_path.read_bytes()
                self._send_bytes(
                    HTTPStatus.OK,
                    "application/pdf",
                    pdf_bytes,
                    {"Content-Disposition": 'attachment; filename="thermdash_audit_report.pdf"'},
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except (ValueError, json.JSONDecodeError, MissingEnergyConstantsError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def create_server(host: str = "127.0.0.1", port: int = 8501) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer for ThermDash UI."""
    return ThreadingHTTPServer((host, port), ThermDashRequestHandler)


def run_server(host: str = "127.0.0.1", port: int = 8501, open_browser: bool = True) -> None:
    """Start the ThermDash web UI server and block until interrupted."""
    server = create_server(host=host, port=port)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print(f"ThermDash Complete UI running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ThermDash UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
