#!/usr/bin/env python3
"""Unified CLI for ThermDash Complete (energy accounting + n8n audit + Law 25 inventory + PDF report)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from token_dashboard.energy_table import lookup
from token_dashboard.inventory import export_inventory_csv, export_inventory_json, inventory_from_n8n
from token_dashboard.n8n_audit import N8nWorkflowAuditor
from token_dashboard.pdf_report import generate_audit_pdf
from token_dashboard.thermodynamics import efficiency


def cmd_calc_energy(args: argparse.Namespace) -> int:
    c = lookup(args.model)
    rep = efficiency(
        n_in=args.tokens_in,
        n_out=args.tokens_out,
        c=c,
        q=args.quality,
        n_reasoning=args.reasoning_tokens,
        rework_turns=args.rework_turns,
        latency_ms=args.latency_ms,
    )
    print(json.dumps({
        "model": c.model,
        "gpu": c.gpu,
        "quantization": c.quantization,
        "measured": c.measured,
        "e_in_joules": round(rep.e_in, 6),
        "e_in_iqr_low_joules": round(rep.e_in_low, 6),
        "e_in_iqr_high_joules": round(rep.e_in_high, 6),
        "w_useful_joules": round(rep.w_useful, 6),
        "eta": round(rep.eta, 4),
        "waste_share": round(rep.waste_share, 4),
        "rework_turns": rep.rework_turns,
        "latency_ms": rep.latency_ms,
    }, indent=2))
    return 0


def cmd_audit_n8n(args: argparse.Namespace) -> int:
    auditor = N8nWorkflowAuditor.from_file(args.workflow)
    exec_data = json.loads(Path(args.execution_log).read_text(encoding="utf-8")) if args.execution_log else None
    report = auditor.audit(
        assumed_in_tokens=args.tokens_in,
        assumed_out_tokens=args.tokens_out,
        execution_data=exec_data,
    )
    print(f"=== ThermDash Complete — n8n Audit: {report.workflow_name} ===")
    print(f"Data Basis: {report.energy_source.upper()} | Nodes: {report.total_nodes}")
    print(f"Est. Energy/Run: {report.estimated_run_joules:.4f} J "
          f"(IQR: {report.estimated_run_joules_low:.4f}–{report.estimated_run_joules_high:.4f} J)")
    print(f"Est. API Cost/Run: ${report.estimated_run_cost_usd:.4f} | "
          f"Est. Monetary Loss/Run: ${report.estimated_monetary_loss_per_run_usd:.4f}\n")

    if args.inventory_out:
        items = inventory_from_n8n(report)
        if args.inventory_out.endswith(".csv"):
            export_inventory_csv(items, args.inventory_out)
        else:
            export_inventory_json(items, args.inventory_out)
        print(f"Inventory written to: {args.inventory_out}")

    if args.mermaid_out:
        Path(args.mermaid_out).write_text(report.mermaid_graph, encoding="utf-8")
        print(f"Mermaid graph written to: {args.mermaid_out}")

    if args.pdf_out:
        generate_audit_pdf(
            report,
            args.pdf_out,
            monthly_runs=args.monthly_runs,
            staff_hourly_rate_usd=args.hourly_rate,
        )
        print(f"Executive PDF report written to: {args.pdf_out}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thermdash", description="ThermDash Complete CLI")
    sub = p.add_subparsers(dest="command", required=True)

    calc = sub.add_parser("calc-energy", help="Compute thermodynamic efficiency with IQR bounds")
    calc.add_argument("--model", required=True)
    calc.add_argument("--in", dest="tokens_in", type=int, required=True)
    calc.add_argument("--out", dest="tokens_out", type=int, required=True)
    calc.add_argument("--reasoning", dest="reasoning_tokens", type=int, default=0)
    calc.add_argument("--quality", type=float, default=None)
    calc.add_argument("--rework-turns", type=int, default=1)
    calc.add_argument("--latency-ms", type=float, default=None)
    calc.set_defaults(func=cmd_calc_energy)

    n8n = sub.add_parser("audit-n8n", help="Audit an n8n workflow JSON for waste, risk, energy, and monetary loss")
    n8n.add_argument("workflow", help="Path to n8n workflow JSON")
    n8n.add_argument("--execution-log", default=None, help="Optional n8n execution JSON for real runtime token counts")
    n8n.add_argument("--in", dest="tokens_in", type=int, default=1000)
    n8n.add_argument("--out", dest="tokens_out", type=int, default=400)
    n8n.add_argument("--monthly-runs", type=int, default=1000)
    n8n.add_argument("--hourly-rate", type=float, default=45.0)
    n8n.add_argument("--inventory-out", default=None, help="Export Law 25 tool inventory (.csv or .json)")
    n8n.add_argument("--mermaid-out", default=None, help="Export Mermaid dependency graph (.mmd)")
    n8n.add_argument("--pdf-out", default=None, help="Export formatted executive PDF report (.pdf)")
    n8n.set_defaults(func=cmd_audit_n8n)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
