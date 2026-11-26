"""Executive PDF report generator for ThermDash Complete (ReportLab).

Displays the automation/AI service name, main estimated monetary loss callout,
LLM features, agent harness details (if applicable), and AI automation / Law 25
review inventory (if applicable).
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
    rework_minutes_per_retry: float = 5.0,
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
        "TDTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#1B263B"), spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "TDSub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#415A77"), spaceAfter=14
    )
    h2_style = ParagraphStyle(
        "TDH2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#1B263B"), spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle("TDBody", parent=styles["Normal"], fontSize=9.5, leading=13)
    hero_label = ParagraphStyle("TDHeroLabel", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#7F1D1D"))
    hero_value = ParagraphStyle("TDHeroVal", parent=styles["Title"], fontSize=24, textColor=colors.HexColor("#991B1B"), alignment=0)

    api_loss_monthly = report.estimated_monetary_loss_per_run_usd * monthly_runs
    rework_cost_per_run = (report.total_rework_runs * rework_minutes_per_retry / 60.0) * staff_hourly_rate_usd
    rework_loss_monthly = rework_cost_per_run * monthly_runs
    total_monthly_loss = api_loss_monthly + rework_loss_monthly

    story = []

    # Header
    story.append(Paragraph(f"AI Efficiency Audit: {report.workflow_name}", title_style))
    source_label = report.energy_source.replace("_", " ").upper()
    story.append(
        Paragraph(
            f"Prepared by Central 73 | Generated: {datetime.now().strftime('%Y-%m-%d')} | Data Basis: <b>{source_label}</b>",
            subtitle_style,
        )
    )

    # Main Dish: Monetary Loss Hero Callout
    hero_data = [
        [Paragraph("ESTIMATED MONTHLY MONETARY LOSS (MAIN FINDING)", hero_label)],
        [Paragraph(f"${total_monthly_loss:,.2f} USD / month", hero_value)],
        [
            Paragraph(
                f"Projected at {monthly_runs:,} runs/month: "
                f"API waste ${api_loss_monthly:,.2f} (${report.estimated_monetary_loss_per_run_usd:.4f}/run) + "
                f"Staff rework ${rework_loss_monthly:,.2f} ({report.total_rework_runs} retry loops/run @ ${staff_hourly_rate_usd:.0f}/hr). "
                f"Total API spend: ${report.estimated_run_cost_usd * monthly_runs:,.2f}/month.",
                body_style,
            )
        ],
    ]
    hero_table = Table(hero_data, colWidths=[7.0 * inch])
    hero_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#DC2626")),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(hero_table)
    story.append(Spacer(1, 10))

    # Section 1: LLM Features & Thermodynamic Profile
    story.append(Paragraph("1. LLM Features & Thermodynamic Profile", h2_style))
    if report.llm_features:
        rows = [["Node", "Model", "GPU / Quant", "Reasoning", "Tokens (In/Out)", "Waste %", "Loss/Run"]]
        for f in report.llm_features:
            rows.append([
                f["node_name"][:20],
                f["model"],
                f"{f['gpu']} / {f['quantization']}",
                "Yes" if f["is_reasoning"] else "No",
                f"{f['tokens_in']} / {f['tokens_out']}",
                f"{f['waste_share'] * 100:.1f}%",
                f"${f['loss_usd']:.4f}",
            ])
        t = Table(rows, colWidths=[1.25 * inch, 1.25 * inch, 0.95 * inch, 0.7 * inch, 1.15 * inch, 0.75 * inch, 0.95 * inch])
        t.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B263B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(t)
    else:
        story.append(Paragraph("No pinned LLM models with energy constants were detected in this workflow.", body_style))

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"<b>Energy per run:</b> {report.estimated_run_joules:.4f} J "
            f"(IQR: {report.estimated_run_joules_low:.4f} - {report.estimated_run_joules_high:.4f} J) | "
            f"<b>Observed latency:</b> {report.total_latency_ms:.0f} ms",
            body_style,
        )
    )

    # Section 2: Agent Harness (if applicable)
    if report.agent_harness_nodes:
        story.append(Paragraph("2. Agent Harness & Orchestration", h2_style))
        for item in report.agent_harness_nodes:
            story.append(Paragraph(f"- {item}", body_style))

    # Section 3: AI Automation & Risk Findings (if applicable)
    story.append(Paragraph("3. AI Automation Diagnostics & Review Flags", h2_style))
    story.append(Paragraph(f"Total workflow nodes scanned: {report.total_nodes}", body_style))
    for flag in report.red_flags[:8]:
        story.append(Paragraph(f"<b>[RED]</b> {flag}", body_style))
    for flag in report.yellow_flags[:6]:
        story.append(Paragraph(f"<b>[YELLOW]</b> {flag}", body_style))
    for mig in report.migrations[:6]:
        story.append(Paragraph(f"<b>[OPTION]</b> {mig}", body_style))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "<i>Scope note: Estimates depend on the stated data basis (runtime log vs. baseline assumptions) "
            "and published model pricing/energy literature. Regulatory flags are engineering discovery indicators "
            "and do not constitute legal, compliance, or guaranteed savings claims.</i>",
            ParagraphStyle("TDFoot", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#64748B")),
        )
    )

    doc.build(story)
    return dest
