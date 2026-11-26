"""Modular n8n workflow auditor synthesized from ThermDash-AutoAudit.

Inspects n8n workflow JSON files for:
- Cloud vs. local AI vendor dependencies and raw httpRequest bypasses
- Structural reliability gaps (missing Error Trigger, hardcoded memory keys, unpinned models)
- PII field hints and Law 25 / GDPR review flags (non-legal engineering indicators)
- Estimated per-run thermodynamic energy footprint when model names are pinned
- Mermaid dependency graph generation and Law 25 tool inventory extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .energy_table import lookup
from .thermodynamics import MissingEnergyConstantsError, efficiency

VENDOR_ALIASES = {
    "lmChatOpenAi": "openai",
    "openAi": "openai",
    "lmChatAnthropic": "anthropic",
    "lmChatOllama": "local_ollama",
    "lmChatLmStudio": "local_lmstudio",
    "telegram": "telegram",
    "telegramTrigger": "telegram",
    "httpRequest": "custom_api",
    "agent": "langchain_agent",
    "memoryBufferWindow": "n8n_memory",
    "set": "native_data",
    "if": "native_logic",
    "switch": "native_logic",
    "baserowTool": "database",
    "gmailTool": "google",
    "googleCalendarTool": "google",
    "cognee": "local_memory",
    "stickyNote": "native_ui",
    "manualTrigger": "native_trigger",
    "chainLlm": "langchain_logic",
    "outputParserStructured": "langchain_logic",
    "scheduleTrigger": "native_trigger",
    "mistralAi": "mistral",
    "formTrigger": "native_trigger",
    "googleSheets": "google",
    "gmail": "google",
    "lmChatGoogleGemini": "google",
    "code": "native_logic",
}

CLOUD_AI_DOMAINS = [
    "api.openai.com",
    "api.anthropic.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "generativelanguage.googleapis.com",
    "api.together.xyz",
    "openrouter.ai",
    "api.groq.com",
]

LOCAL_REPLACEMENTS = {
    "lmChatOpenAi": "Local 8B-Class LLM (e.g., Llama 3/Mistral via vLLM/Llama.cpp)",
    "lmChatAnthropic": "Local 8B-Class LLM (e.g., Llama 3/Mistral via vLLM/Llama.cpp)",
    "openAi_audio": "Local Audio Transcription (e.g., Whisper.cpp)",
    "openAi_text": "Local 8B-Class LLM (e.g., Llama 3/Mistral via vLLM/Llama.cpp)",
    "agent": "Sovereign Python Orchestration (e.g., local workflow engine)",
    "memoryBufferWindow": "Persistent Local Memory (e.g., Self-hosted SQLite/PostgreSQL)",
    "baserowTool": "Self-hosted CRM/Database (e.g., Baserow Self-Hosted or Local SQL)",
    "gmailTool": "Self-hosted Email (e.g., Postal or Listmonk)",
    "googleCalendarTool": "Self-hosted Calendar (e.g., Cal.com or Nextcloud Calendar)",
    "mistralAi": "Local 8B-Class LLM (e.g., Llama 3/Mistral via vLLM/Llama.cpp)",
    "googleSheets": "Self-hosted Spreadsheet/DB (e.g., NocoDB or Baserow)",
    "gmail": "Self-hosted Email (e.g., Postal or Listmonk)",
    "lmChatGoogleGemini": "Local 8B-Class LLM (e.g., Llama 3/Mistral via vLLM/Llama.cpp)",
}

REGULATORY_HINTS = {
    "cross_border": "[Review: Law 25 Art. 17 — potential cross-border transfer; verify PIA & contract]",
    "google": "[Review: Law 25 Art. 17 / Art. 12 — external cloud transfer; verify disclosure]",
    "httpRequest": "[Review: Law 25 Art. 17 — undeclared external AI API endpoint]",
    "hardcoded_session": "[Review: Law 25 Art. 13 — verify retention/deletion protocol]",
    "unpinned_model": "[Review: Model version unpinned — risk of silent cost/behavior drift]",
    "no_error_trigger": "[Review: Reliability gap — no Error Trigger for failed automated runs]",
}

PII_FIELD_HINTS = ["email", "name", "phone", "dob", "address", "postal", "birth", "ssn", "sin", "patient"]


@dataclass
class WorkflowAuditReport:
    workflow_name: str
    total_nodes: int
    red_flags: list[str] = field(default_factory=list)
    yellow_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)
    migrations: list[str] = field(default_factory=list)
    detected_nodes: list[dict[str, Any]] = field(default_factory=list)
    estimated_run_joules: float = 0.0
    estimated_run_joules_low: float = 0.0
    estimated_run_joules_high: float = 0.0
    mermaid_graph: str = ""


class N8nWorkflowAuditor:
    """Parses and audits an n8n workflow JSON export."""

    def __init__(self, workflow_data: dict[str, Any]):
        self.workflow = workflow_data
        self.nodes = workflow_data.get("nodes", [])
        self.connections = workflow_data.get("connections", {})

    @classmethod
    def from_file(cls, filepath: str | Path) -> "N8nWorkflowAuditor":
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return cls(data)

    def _safe_node_name(self, node: dict[str, Any], index: int) -> str:
        return node.get("name") or f"UnnamedNode_{index}"

    def generate_mermaid_graph(self) -> str:
        mermaid = [
            "graph TD",
            "  classDef cloud fill:#ffebee,stroke:#cc0000,stroke-width:2px,color:#900",
            "  classDef local fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000",
            "  classDef neutral fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,color:#000",
        ]
        node_styles: dict[str, dict[str, str]] = {}

        for i, node in enumerate(self.nodes):
            node_name = self._safe_node_name(node, i)
            safe_id = f"N{i}"
            raw_type = node.get("type", "").split(".")[-1]
            vendor = VENDOR_ALIASES.get(raw_type, "unknown")

            if vendor in ("openai", "anthropic", "google", "telegram", "mistral"):
                style = "cloud"
                label = f"{node_name}\\n[cloud: {vendor}]"
            elif vendor in (
                "local_ollama", "local_lmstudio", "n8n_memory", "local_memory",
                "native_logic", "native_trigger", "native_data", "native_ui",
                "langchain_logic", "langchain_agent", "database",
            ):
                style = "local"
                label = f"{node_name}\\n[local: {vendor}]"
            else:
                style = "neutral"
                label = f"{node_name}\\n[other: {vendor}]"

            mermaid.append(f'  {safe_id}["{label}"]')
            node_styles[node_name] = {"id": safe_id, "style": style}

        for source_name, output_data in self.connections.items():
            if source_name not in node_styles:
                continue
            source_id = node_styles[source_name]["id"]
            for output_type, targets in output_data.items():
                for target_group in targets:
                    for target in target_group:
                        target_name = target.get("node")
                        if target_name in node_styles:
                            target_id = node_styles[target_name]["id"]
                            mermaid.append(f"  {source_id} -->|{output_type}| {target_id}")

        for data in node_styles.values():
            mermaid.append(f"  class {data['id']} {data['style']}")

        return "\n".join(mermaid)

    def audit(
        self,
        assumed_in_tokens: int = 1000,
        assumed_out_tokens: int = 400,
    ) -> WorkflowAuditReport:
        report = WorkflowAuditReport(
            workflow_name=self.workflow.get("name", "Unknown Workflow"),
            total_nodes=len(self.nodes),
            mermaid_graph=self.generate_mermaid_graph(),
        )

        has_error_trigger = any("error" in n.get("type", "").lower() for n in self.nodes)
        if not has_error_trigger:
            report.red_flags.append(
                f"No Error Trigger detected; workflow may fail silently. {REGULATORY_HINTS['no_error_trigger']}"
            )

        ai_vendors: set[str] = set()

        for i, node in enumerate(self.nodes):
            raw_type = node.get("type", "").split(".")[-1]
            node_name = self._safe_node_name(node, i)
            vendor = VENDOR_ALIASES.get(raw_type, "unknown")
            params = node.get("parameters", {})
            model_raw = params.get("model", "")
            model_name = model_raw.get("value", "") if isinstance(model_raw, dict) else str(model_raw)

            report.detected_nodes.append({
                "node_name": node_name,
                "node_type": raw_type,
                "vendor": vendor,
                "model": model_name,
                "is_cloud": vendor in ("openai", "anthropic", "google", "mistral", "telegram"),
            })

            if "memory" in node.get("type", "").lower():
                key = params.get("sessionKey", "")
                if key and "$" not in str(key):
                    report.red_flags.append(
                        f"Hardcoded memory sessionKey '{key}' in '{node_name}'. {REGULATORY_HINTS['hardcoded_session']}"
                    )

            if raw_type == "httpRequest":
                url = str(params.get("url", ""))
                for domain in CLOUD_AI_DOMAINS:
                    if domain in url:
                        report.red_flags.append(
                            f"Node '{node_name}' calls '{domain}' via raw HTTP. {REGULATORY_HINTS['httpRequest']}"
                        )
                        break

            lookup_type = raw_type
            if raw_type == "openAi":
                lookup_type = "openAi_audio" if params.get("resource") == "audio" else "openAi_text"

            if lookup_type in LOCAL_REPLACEMENTS:
                report.migrations.append(f"'{node_name}' -> Consider {LOCAL_REPLACEMENTS[lookup_type]}")

            if vendor == "google":
                report.red_flags.append(
                    f"Node '{node_name}' sends data to Google services. {REGULATORY_HINTS['google']}"
                )

            if vendor in ("openai", "anthropic", "mistral", "google"):
                if raw_type in ("lmChatGoogleGemini", "lmChatOpenAi", "openAi", "lmChatAnthropic", "mistralAi"):
                    ai_vendors.add(vendor)
                if not model_name and raw_type not in ("googleCalendarTool", "googleSheets", "gmail", "gmailTool"):
                    report.yellow_flags.append(
                        f"{vendor.upper()} model unpinned in '{node_name}'. {REGULATORY_HINTS['unpinned_model']}"
                    )
                elif model_name:
                    try:
                        c = lookup(model_name)
                        eff = efficiency(assumed_in_tokens, assumed_out_tokens, c)
                        report.estimated_run_joules += eff.e_in
                        report.estimated_run_joules_low += eff.e_in_low
                        report.estimated_run_joules_high += eff.e_in_high
                    except MissingEnergyConstantsError:
                        report.yellow_flags.append(
                            f"Model '{model_name}' in '{node_name}' not found in energy.json; energy unestimated."
                        )

            if vendor in ("openai", "anthropic"):
                params_str = json.dumps(params).lower()
                for hint in PII_FIELD_HINTS:
                    if hint in params_str:
                        report.red_flags.append(
                            f"PII field hint '{hint}' detected in cloud node '{node_name}'. {REGULATORY_HINTS['cross_border']}"
                        )
                        break

            if vendor in ("local_ollama", "local_lmstudio", "local_memory"):
                report.green_flags.append(
                    f"Node '{node_name}' uses local infrastructure (reduces cross-border transfer exposure; not a legal determination)."
                )

        if len(ai_vendors) == 1:
            v = next(iter(ai_vendors)).upper()
            report.yellow_flags.append(f"Single-vendor dependency on {v} with no fallback configured.")
        elif len(ai_vendors) > 1:
            report.yellow_flags.append(
                f"Mixed cloud AI stack ({', '.join(sorted(ai_vendors))}) — review for tool duplication."
            )

        report.red_flags = sorted(set(report.red_flags))
        report.yellow_flags = sorted(set(report.yellow_flags))
        report.green_flags = sorted(set(report.green_flags))
        return report
