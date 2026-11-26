"""Modular n8n workflow auditor with runtime execution-log ingestion and EcoLogits integration.

Separates:
1. Financial operating cost (from pricing.json + runtime logs or explicit baseline assumptions)
2. Environmental estimates (from EcoLogits; unsupported models marked 'not_estimated', never 0.0)
3. Operational / governance review flags (non-legal engineering indicators)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .environmental import estimate_cloud_impact

_DEFAULT_PRICING_JSON = Path(__file__).parent.parent / "pricing.json"

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
    "api.vapi.ai",
    "api.retellai.com",
    "api.elevenlabs.io",
    "api.bland.ai",
]

LOCAL_REPLACEMENTS = {
    "lmChatOpenAi": "Candidate: smaller API model or locally hosted open-weight LLM (validate quality first)",
    "lmChatAnthropic": "Candidate: smaller API model or locally hosted open-weight LLM (validate quality first)",
    "openAi_audio": "Candidate: local audio transcription (e.g., Whisper.cpp)",
    "openAi_text": "Candidate: smaller API model or locally hosted open-weight LLM",
    "agent": "Candidate: deterministic script or narrower local orchestration",
    "memoryBufferWindow": "Candidate: self-hosted SQLite/PostgreSQL state",
    "baserowTool": "Candidate: self-hosted Baserow or local SQL",
    "gmailTool": "Candidate: self-hosted SMTP / Postal / Listmonk",
    "googleCalendarTool": "Candidate: self-hosted Cal.com / Nextcloud Calendar",
    "mistralAi": "Candidate: locally hosted open-weight LLM",
    "googleSheets": "Candidate: self-hosted NocoDB / Baserow / SQLite",
    "gmail": "Candidate: self-hosted SMTP / Postal / Listmonk",
    "lmChatGoogleGemini": "Candidate: smaller API model or locally hosted open-weight LLM",
}

REGULATORY_HINTS = {
    "cross_border": "[Review: Law 25 Art. 17 — potential cross-border transfer; verify PIA & contract]",
    "google": "[Review: Law 25 Art. 17 / Art. 12 — external cloud transfer; verify disclosure]",
    "httpRequest": "[Review: Law 25 Art. 17 — external AI/voice API endpoint]",
    "hardcoded_session": "[Review: Law 25 Art. 13 — verify retention/deletion protocol]",
    "unpinned_model": "[Review: Model version unpinned — risk of silent cost/behavior drift]",
    "no_error_trigger": "[Review: Reliability gap — no Error Trigger for failed automated runs]",
}

PII_FIELD_HINTS = ["email", "name", "phone", "dob", "address", "postal", "birth", "ssn", "sin", "patient"]


def _load_pricing(path: Path | None = None) -> dict[str, dict[str, float]]:
    p = path or _DEFAULT_PRICING_JSON
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("models", {})


def _estimate_token_cost_usd(
    model: str, tokens_in: int, tokens_out: int, pricing: dict[str, dict[str, float]]
) -> float | None:
    rates = pricing.get(model)
    if not rates:
        for k, v in pricing.items():
            if model.lower().startswith(k.lower()):
                rates = v
                break
    if not rates:
        return None
    return (tokens_in / 1_000_000.0) * rates["input"] + (tokens_out / 1_000_000.0) * rates["output"]


def _extract_model_name(params: dict[str, Any]) -> str:
    for key in ("model", "modelId"):
        raw = params.get(key)
        if isinstance(raw, dict):
            val = str(raw.get("value", "")).strip()
            if val:
                return val
        elif isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def extract_runtime_metrics(execution_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract per-node token counts, latency_ms, and run count from n8n execution JSON."""
    run_data = execution_data.get("data", {}).get("resultData", {}).get("runData", {})
    metrics: dict[str, dict[str, Any]] = {}

    for node_name, runs in run_data.items():
        if not isinstance(runs, list) or not runs:
            continue
        tokens_in, tokens_out, latency_ms = 0, 0, 0.0
        has_tokens = False

        for run in runs:
            latency_ms += float(run.get("executionTime", 0.0) or 0.0)
            data_block = run.get("data", {})
            for channel in ("ai_languageModel", "main"):
                for group in data_block.get(channel, []) or []:
                    for item in group or []:
                        j = item.get("json", {}) if isinstance(item, dict) else {}
                        usage = j.get("tokenUsage") or j.get("usage") or {}
                        p = usage.get("promptTokens", usage.get("prompt_tokens", usage.get("input_tokens")))
                        c = usage.get("completionTokens", usage.get("completion_tokens", usage.get("output_tokens")))
                        if p is not None or c is not None:
                            tokens_in += int(p or 0)
                            tokens_out += int(c or 0)
                            has_tokens = True

        metrics[node_name] = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "runs": len(runs),
            "has_tokens": has_tokens,
        }
    return metrics


@dataclass
class WorkflowAuditReport:
    workflow_name: str
    total_nodes: int
    sticky_notes_ignored: int = 0
    cost_source: str = "baseline_estimate"  # "runtime_log", "mixed", or "baseline_estimate"
    red_flags: list[str] = field(default_factory=list)
    yellow_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)
    migrations: list[str] = field(default_factory=list)
    detected_nodes: list[dict[str, Any]] = field(default_factory=list)
    llm_features: list[dict[str, Any]] = field(default_factory=list)
    environmental_estimates: list[dict[str, Any]] = field(default_factory=list)
    agent_harness_nodes: list[str] = field(default_factory=list)
    estimated_run_cost_usd: float | None = None
    total_latency_ms: float = 0.0
    extra_model_calls: int = 0
    mermaid_graph: str = ""


class N8nWorkflowAuditor:
    """Parses and audits an n8n workflow JSON export, optionally enriched with runtime logs."""

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
            raw_type = node.get("type", "").split(".")[-1]
            if raw_type == "stickyNote":
                continue
            node_name = self._safe_node_name(node, i)
            safe_id = f"N{i}"
            vendor = VENDOR_ALIASES.get(raw_type, "unknown")

            if vendor in ("openai", "anthropic", "google", "telegram", "mistral"):
                style = "cloud"
                label = f"{node_name}\\n[cloud: {vendor}]"
            elif vendor in (
                "local_ollama", "local_lmstudio", "n8n_memory", "local_memory",
                "native_logic", "native_trigger", "native_data",
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
        execution_data: dict[str, Any] | None = None,
    ) -> WorkflowAuditReport:
        runtime_map = extract_runtime_metrics(execution_data) if execution_data else {}
        pricing = _load_pricing()

        active_nodes = [n for n in self.nodes if n.get("type", "").split(".")[-1] != "stickyNote"]
        sticky_count = len(self.nodes) - len(active_nodes)

        report = WorkflowAuditReport(
            workflow_name=self.workflow.get("name", "Unknown Workflow"),
            total_nodes=len(active_nodes),
            sticky_notes_ignored=sticky_count,
            mermaid_graph=self.generate_mermaid_graph(),
        )

        has_error_trigger = any("error" in n.get("type", "").lower() for n in active_nodes)
        if not has_error_trigger:
            report.red_flags.append(
                f"No Error Trigger detected; workflow may fail silently. {REGULATORY_HINTS['no_error_trigger']}"
            )

        ai_vendors: set[str] = set()
        runtime_hits = 0
        baseline_hits = 0
        total_cost = 0.0
        any_cost_computed = False

        for i, node in enumerate(active_nodes):
            raw_type = node.get("type", "").split(".")[-1]
            node_name = self._safe_node_name(node, i)
            vendor = VENDOR_ALIASES.get(raw_type, "unknown")
            params = node.get("parameters", {})
            model_name = _extract_model_name(params)

            if vendor in ("langchain_agent", "langchain_logic", "n8n_memory", "local_memory"):
                report.agent_harness_nodes.append(f"{node_name} ({raw_type})")

            is_cloud_http = False
            if raw_type == "httpRequest":
                url = str(params.get("url", ""))
                for domain in CLOUD_AI_DOMAINS:
                    if domain in url:
                        is_cloud_http = True
                        ai_vendors.add(domain)
                        report.red_flags.append(
                            f"Node '{node_name}' calls external AI/voice API '{domain}' via HTTP. {REGULATORY_HINTS['httpRequest']}"
                        )
                        break

            report.detected_nodes.append({
                "node_name": node_name,
                "node_type": raw_type,
                "vendor": vendor,
                "model": model_name,
                "is_cloud": is_cloud_http or vendor in ("openai", "anthropic", "google", "mistral", "telegram"),
            })

            if "memory" in node.get("type", "").lower():
                key = params.get("sessionKey", "")
                if key and "$" not in str(key):
                    report.red_flags.append(
                        f"Hardcoded memory sessionKey '{key}' in '{node_name}'. {REGULATORY_HINTS['hardcoded_session']}"
                    )

            lookup_type = raw_type
            if raw_type == "openAi":
                lookup_type = "openAi_audio" if params.get("resource") == "audio" else "openAi_text"

            if lookup_type in LOCAL_REPLACEMENTS:
                report.migrations.append(f"'{node_name}' -> {LOCAL_REPLACEMENTS[lookup_type]}")

            if vendor == "google":
                report.red_flags.append(
                    f"Node '{node_name}' sends data to Google services. {REGULATORY_HINTS['google']}"
                )

            if vendor in ("openai", "anthropic", "mistral", "google", "local_ollama", "local_lmstudio"):
                if raw_type in ("lmChatGoogleGemini", "lmChatOpenAi", "openAi", "lmChatAnthropic", "mistralAi"):
                    ai_vendors.add(vendor)
                if not model_name and raw_type not in ("googleCalendarTool", "googleSheets", "gmail", "gmailTool"):
                    report.yellow_flags.append(
                        f"{vendor.upper()} model unpinned in '{node_name}'. {REGULATORY_HINTS['unpinned_model']}"
                    )
                elif model_name:
                    rt = runtime_map.get(node_name)
                    if rt and rt["has_tokens"]:
                        n_in, n_out = rt["tokens_in"], rt["tokens_out"]
                        runs = max(1, rt["runs"])
                        latency = rt["latency_ms"]
                        runtime_hits += 1
                        basis = "runtime_log"
                    else:
                        n_in, n_out = assumed_in_tokens, assumed_out_tokens
                        runs = 1
                        latency = rt["latency_ms"] if rt else None
                        baseline_hits += 1
                        basis = "baseline_estimate"

                    if latency:
                        report.total_latency_ms += latency
                    if runs > 1:
                        report.extra_model_calls += runs - 1

                    cost_usd = _estimate_token_cost_usd(model_name, n_in, n_out, pricing)
                    if cost_usd is not None:
                        total_cost += cost_usd
                        any_cost_computed = True
                    else:
                        report.yellow_flags.append(
                            f"Model '{model_name}' in '{node_name}' not found in pricing.json; API cost not quantified."
                        )

                    # Separate environmental estimate via EcoLogits (never merged into financial cost)
                    eco = estimate_cloud_impact(
                        provider=vendor,
                        model_name=model_name,
                        output_tokens=n_out,
                        request_latency_s=(latency / 1000.0) if latency else 1.0,
                    )
                    report.environmental_estimates.append({
                        "node_name": node_name,
                        **asdict(eco),
                    })

                    report.llm_features.append({
                        "node_name": node_name,
                        "vendor": vendor,
                        "model": model_name,
                        "tokens_in": n_in,
                        "tokens_out": n_out,
                        "runs": runs,
                        "basis": basis,
                        "cost_usd": cost_usd,
                        "env_status": eco.status,
                        "env_energy_kwh": eco.energy_kwh,
                        "env_gwp_kgco2eq": eco.gwp_kgco2eq,
                    })

            if vendor in ("openai", "anthropic") or is_cloud_http:
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

        report.estimated_run_cost_usd = total_cost if any_cost_computed else None
        if runtime_hits > 0 and baseline_hits == 0:
            report.cost_source = "runtime_log"
        elif runtime_hits > 0 and baseline_hits > 0:
            report.cost_source = "mixed"
        else:
            report.cost_source = "baseline_estimate"

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
