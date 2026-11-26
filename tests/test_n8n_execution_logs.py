"""Tests for n8n runtime execution-log ingestion, monetary loss calculation, and PDF output."""

from pathlib import Path

from token_dashboard.n8n_audit import N8nWorkflowAuditor, extract_runtime_metrics
from token_dashboard.pdf_report import generate_audit_pdf


SAMPLE_WORKFLOW = {
    "name": "Support Triage Agent",
    "nodes": [
        {
            "name": "AI Agent",
            "type": "@n8n/n8n-nodes-langchain.agent",
            "parameters": {},
        },
        {
            "name": "OpenAI Chat Model",
            "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
            "parameters": {"model": "gpt-4o", "prompt": "Summarize customer email"},
        },
        {
            "name": "Fallback Anthropic",
            "type": "@n8n/n8n-nodes-langchain.lmChatAnthropic",
            "parameters": {"model": "claude-3-5-sonnet"},
        },
    ],
    "connections": {},
}


def test_extract_langchain_token_usage_and_retries():
    exec_json = {
        "data": {
            "resultData": {
                "runData": {
                    "OpenAI Chat Model": [
                        {
                            "executionTime": 1200,
                            "data": {
                                "ai_languageModel": [
                                    [{"json": {"tokenUsage": {"promptTokens": 800, "completionTokens": 200}}}]
                                ]
                            },
                        },
                        {
                            "executionTime": 900,
                            "data": {
                                "ai_languageModel": [
                                    [{"json": {"tokenUsage": {"promptTokens": 900, "completionTokens": 250}}}]
                                ]
                            },
                        },
                    ]
                }
            }
        }
    }
    m = extract_runtime_metrics(exec_json)
    assert m["OpenAI Chat Model"]["tokens_in"] == 1700
    assert m["OpenAI Chat Model"]["tokens_out"] == 450
    assert m["OpenAI Chat Model"]["latency_ms"] == 2100.0
    assert m["OpenAI Chat Model"]["runs"] == 2
    assert m["OpenAI Chat Model"]["has_tokens"] is True


def test_audit_mixed_runtime_and_baseline_plus_pdf(tmp_path: Path):
    exec_json = {
        "data": {
            "resultData": {
                "runData": {
                    "OpenAI Chat Model": [
                        {
                            "executionTime": 1100,
                            "data": {
                                "ai_languageModel": [
                                    [{"json": {"tokenUsage": {"promptTokens": 1500, "completionTokens": 500}}}]
                                ]
                            },
                        },
                        {
                            "executionTime": 1000,
                            "data": {
                                "ai_languageModel": [
                                    [{"json": {"tokenUsage": {"promptTokens": 1500, "completionTokens": 500}}}]
                                ]
                            },
                        },
                    ]
                }
            }
        }
    }
    report = N8nWorkflowAuditor(SAMPLE_WORKFLOW).audit(execution_data=exec_json)
    assert report.energy_source == "mixed"
    assert report.total_rework_runs == 1
    assert report.estimated_run_cost_usd > 0.0
    assert report.estimated_monetary_loss_per_run_usd > 0.0
    assert len(report.agent_harness_nodes) == 1

    pdf_file = tmp_path / "audit_report.pdf"
    generate_audit_pdf(report, pdf_file, monthly_runs=500, staff_hourly_rate_usd=50.0)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1000
