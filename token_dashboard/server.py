"""Local-first web server for ThermDash Complete UI (no external web framework required)."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile

from .inventory import export_inventory_csv, inventory_from_n8n
from .n8n_audit import N8nWorkflowAuditor
from .pdf_report import generate_audit_pdf

_WEB_DIR = Path(__file__).parent.parent / "web"


class ThermDashHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (_WEB_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        self._send(404, b"Not Found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            workflow = payload["workflow"]
            exec_log = payload.get("execution_log")
            monthly_runs = int(payload.get("monthly_runs", 1000))
            hourly_rate = float(payload.get("hourly_rate", 45.0))
            in_tok = int(payload.get("assumed_in_tokens", 1000))
            out_tok = int(payload.get("assumed_out_tokens", 400))

            report = N8nWorkflowAuditor(workflow).audit(
                assumed_in_tokens=in_tok,
                assumed_out_tokens=out_tok,
                execution_data=exec_log,
            )

            if self.path == "/api/audit":
                api_loss_mo = report.estimated_monetary_loss_per_run_usd * monthly_runs
                rework_loss_mo = (report.total_rework_runs * 5.0 / 60.0) * hourly_rate * monthly_runs
                data = asdict(report)
                data["monthly_runs"] = monthly_runs
                data["api_loss_monthly_usd"] = round(api_loss_mo, 2)
                data["rework_loss_monthly_usd"] = round(rework_loss_mo, 2)
                data["total_monthly_loss_usd"] = round(api_loss_mo + rework_loss_mo, 2)
                self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                return

            if self.path == "/api/audit/inventory.csv":
                csv_text = export_inventory_csv(inventory_from_n8n(report))
                self._send(200, csv_text.encode("utf-8"), "text/csv; charset=utf-8", "law25_inventory.csv")
                return

            if self.path == "/api/audit/pdf":
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                generate_audit_pdf(report, tmp_path, monthly_runs=monthly_runs, staff_hourly_rate_usd=hourly_rate)
                pdf_bytes = tmp_path.read_bytes()
                tmp_path.unlink(missing_ok=True)
                self._send(200, pdf_bytes, "application/pdf", "thermdash_audit_report.pdf")
                return

            self._send(404, b"Unknown endpoint", "text/plain")
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = HTTPServer((host, port), ThermDashHandler)
    print(f"ThermDash Complete UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
