"""Tests for n8n runtime execution-log extraction, cost separation, and PDF generation."""

from pathlib import Path

from token_dashboard.n8n_audit import N8nWorkflowAuditor, extract_runtime_metrics
from token_dashboard.pdf_report import generate_audit_pdf


def _sample_workflow():
    return {
        "name": "Lead Intake Automation",
        "nodes": [
            {
                "name": "OpenAI Chat Model",
                "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
                "parameters": {"model": {"value": "gpt-4o-mini"}},
            },
            {
                "name": "AI Agent",
                "type": "@n8n/n8n-nodes-langchain.agent",
                "parameters": {},
            },
            {
                "name": "Note",
                "type": "n8n-nodes-base.stickyNote",
                "parameters": {"content": "Docs"},
            },
        ],
        "connections": {
            "OpenAI Chat Model": {
                "ai_languageModel": [[{"node": "AI Agent", "type": "ai_languageModel", "index": 0}]]
            }
        },
    }


def _sample_execution():
    return {
        "data": {
            "resultData": {
                "runData": {
                    "OpenAI Chat Model": [
                        {
                            "executionTime": 1200,
                            "data": {
                                "ai_languageModel": [
                                    [
                                        {
                                            "json": {
                                                "tokenUsage": {
                                                    "promptTokens": 1500,
                                                    "completionTokens": 400,
                                                }
                                            }
                                        }
                                    ]
                                ]
                            },
                        },
                        {
                            "executionTime": 900,
                            "data": {
                                "ai_languageModel": [
                                    [
                                        {
                                            "json": {
                                                "tokenUsage": {
                                                    "promptTokens": 1700,
                                                    "completionTokens": 350,
                                                }
                                            }
                                        }
                                    ]
                                ]
                            },
                        },
                    ]
                }
            }
        }
    }


def test_extract_runtime_metrics():
    metrics = extract_runtime_metrics(_sample_execution())
    m = metrics["OpenAI Chat Model"]
    assert m["has_tokens"] is True
    assert m["tokens_in"] == 3200
    assert m["tokens_out"] == 750
    assert m["runs"] == 2
    assert m["latency_ms"] == 2100.0


def test_audit_uses_runtime_logs_and_separates_environment():
    auditor = N8nWorkflowAuditor(_sample_workflow())
    report = auditor.audit(execution_data=_sample_execution())
    assert report.cost_source == "runtime_log"
    assert report.extra_model_calls == 1
    assert report.total_latency_ms == 2100.0
    assert report.estimated_run_cost_usd is not None
    assert report.estimated_run_cost_usd > 0.0
    assert len(report.environmental_estimates) == 1
    assert report.environmental_estimates[0]["status"] in ("estimated", "not_estimated")


def test_generate_audit_pdf(tmp_path: Path):
    auditor = N8nWorkflowAuditor(_sample_workflow())
    report = auditor.audit(execution_data=_sample_execution())
    pdf_file = generate_audit_pdf(report, tmp_path / "audit.pdf", monthly_runs=500)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1500
