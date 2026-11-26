"""Executive PDF report generator for ThermDash Complete (ReportLab).

Separates financial operating cost from EcoLogits/CodeCarbon environmental estimates
and explicitly labels missing or unmeasured data as 'Not quantified' / 'Not estimated'.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .n8n_audit import WorkflowAuditReport


def generate_audit_pdf(
    report: WorkflowAuditReport,
    out_path: str | Path,
    monthly_runs: int = 1000,
    staff_hourly_rate_usd: float = 45.0,
) -> Path:
    """Generate a professionally formatted executive PDF audit report."""
    dest = Path(out_path)
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TDTitle", parent=styles["Title"], fontSize=19, textColor=colors.HexColor("#1B263B"), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "TDSub", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#415A77"), spaceAfter=14
    )
    h2_style = ParagraphStyle(
        "TDH2", parent=styles["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1B263B"), spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle("TDBody", parent=styles["Normal"], fontSize=9.5, leading=13)
    hero_label = ParagraphStyle("TDHeroLabel", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#1E293B"))
    hero_value = ParagraphStyle("TDHeroVal", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#0F172A"), alignment=0)

    story = []
    story.append(Paragraph(f"AI Workflow Assessment: {report.workflow_name}", title_style))
    source_label = report.cost_source.replace("_", " ").upper()
    story.append(
        Paragraph(
            f"Prepared by Central 73 | Generated: {datetime.now().strftime('%Y-%m-%d')} | Cost Basis: <b>{source_label}</b>",
            subtitle_style,
        )
    )

    # Financial Summary (strictly separated from environmental estimates)
    if report.estimated_run_cost_usd is not None:
        monthly_api_cost = report.estimated_run_cost_usd * monthly_runs
        headline = f"${monthly_api_cost:,.2f} USD / month (Estimated API Spend)"
        detail = (
            f"Projected at {monthly_runs:,} runs/month (${report.estimated_run_cost_usd:.4f}/run). "
            f"Extra model calls observed: {report.extra_model_calls}/run (verify whether retries or required steps). "
            f"Staff correction cost is not automatically inferred; use actual observed minutes @ ${staff_hourly_rate_usd:.0f}/hr."
        )
    else:
        headline = "Financial Spend: Not Yet Quantified"
        detail = "No pinned models with pricing data were found, or external service invoices are required."

    hero_table = Table(
        [
            [Paragraph("OPERATING COST SUMMARY", hero_label)],
            [Paragraph(headline, hero_value)],
            [Paragraph(detail, body_style)],
        ],
        colWidths=[7.0 * inch],
    )
    hero_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 1.25, colors.HexColor("#475569")),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(hero_table)
    story.append(Spacer(1, 10))

    # Section 1: LLM Inventory & Operating Cost
    story.append(Paragraph("1. LLM Nodes & Token Cost Breakdown", h2_style))
    if report.llm_features:
        rows = [["Node", "Model", "Basis", "Tokens (In/Out)", "Runs", "API Cost/Run", "EcoLogits"]]
        for f in report.llm_features:
            cost_str = f"${f['cost_usd']:.4f}" if f["cost_usd"] is not None else "Not quantified"
            env_str = f"{f['env_energy_kwh']:.5f} kWh" if f["env_status"] == "estimated" and f["env_energy_kwh"] is not None else "Not estimated"
            rows.append([
                f["node_name"][:18],
                f["model"],
                f["basis"],
                f"{f['tokens_in']} / {f['tokens_out']}",
                str(f["runs"]),
                cost_str,
                env_str,
            ])
        t = Table(rows, colWidths=[1.15 * inch, 1.25 * inch, 0.95 * inch, 1.05 * inch, 0.5 * inch, 1.0 * inch, 1.1 * inch])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B263B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(t)
    else:
        story.append(Paragraph("No pinned LLM models detected in this workflow.", body_style))

    # Section 2: Agent Harness
    if report.agent_harness_nodes:
        story.append(Paragraph("2. Agent Harness & Orchestration", h2_style))
        for item in report.agent_harness_nodes:
            story.append(Paragraph(f"- {item}", body_style))

    # Section 3: AI Automation & Governance Flags
    story.append(Paragraph("3. AI Automation Diagnostics & Review Flags", h2_style))
    story.append(
        Paragraph(
            f"Active nodes: {report.total_nodes} (sticky notes ignored: {report.sticky_notes_ignored}) | "
            f"Observed latency: {report.total_latency_ms:.0f} ms",
            body_style,
        )
    )
    for flag in report.red_flags[:8]:
        story.append(Paragraph(f"<b>[RED]</b> {flag}", body_style))
    for flag in report.yellow_flags[:6]:
        story.append(Paragraph(f"<b>[YELLOW]</b> {flag}", body_style))
    for mig in report.migrations[:6]:
        story.append(Paragraph(f"<b>[OPTION]</b> {mig}", body_style))

    # Appendix: Environmental Methodology (EcoLogits / CodeCarbon)
    story.append(Paragraph("4. Environmental Appendix (EcoLogits & CodeCarbon)", h2_style))
    story.append(
        Paragraph(
            "Cloud environmental estimates use EcoLogits when the provider/model is supported; unsupported models are "
            "marked 'Not estimated' rather than zero. Local replacement measurements should be captured with CodeCarbon "
            "(CPU/GPU/RAM) or a wall power meter. Environmental figures are never converted into financial savings.",
            body_style,
        )
    )

    doc.build(story)
    return dest
