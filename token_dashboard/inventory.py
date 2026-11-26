"""Law 25 / Governance AI Tool Inventory exporter.

Aggregates detected AI tools from n8n workflows and/or local chat logs into
structured CSV or JSON inventories (tool, vendor, model, cloud_dependency, source).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Iterable

from .n8n_audit import WorkflowAuditReport


@dataclass(frozen=True)
class InventoryItem:
    source_type: str  # e.g., "n8n_workflow" or "chat_log"
    source_name: str
    tool_or_node: str
    vendor: str
    model: str
    is_cloud: bool
    review_note: str = ""


def inventory_from_n8n(report: WorkflowAuditReport) -> list[InventoryItem]:
    items: list[InventoryItem] = []
    for node in report.detected_nodes:
        if node["vendor"] in ("native_logic", "native_trigger", "native_data", "native_ui"):
            continue
        note = "Verify Law 25 Art. 17 cross-border transfer & PIA" if node["is_cloud"] else "Local / internal node"
        items.append(
            InventoryItem(
                source_type="n8n_workflow",
                source_name=report.workflow_name,
                tool_or_node=node["node_name"],
                vendor=node["vendor"],
                model=node["model"] or "unpinned",
                is_cloud=bool(node["is_cloud"]),
                review_note=note,
            )
        )
    return items


def export_inventory_json(items: Iterable[InventoryItem], out_path: str | Path | None = None) -> str:
    payload = json.dumps([asdict(i) for i in items], indent=2)
    if out_path:
        Path(out_path).write_text(payload, encoding="utf-8")
    return payload


def export_inventory_csv(items: Iterable[InventoryItem], out_path: str | Path | None = None) -> str:
    buf = io.StringIO()
    fieldnames = ["source_type", "source_name", "tool_or_node", "vendor", "model", "is_cloud", "review_note"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        writer.writerow(asdict(item))
    content = buf.getvalue()
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    return content
