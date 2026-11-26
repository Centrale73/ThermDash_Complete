"""Tests for core n8n audit diagnostics and Mermaid graph output."""

from token_dashboard.n8n_audit import N8nWorkflowAuditor


def test_auditor_flags_cloud_and_missing_error_trigger():
    wf = {
        "name": "Patient Triage",
        "nodes": [
            {
                "name": "Triage LLM",
                "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
                "parameters": {"model": {"value": "gpt-4o"}, "prompt": "patient email and phone"},
            },
            {
                "name": "Sheet Log",
                "type": "n8n-nodes-base.googleSheets",
                "parameters": {},
            },
        ],
        "connections": {
            "Triage LLM": {"main": [[{"node": "Sheet Log", "type": "main", "index": 0}]]}
        },
    }
    report = N8nWorkflowAuditor(wf).audit()
    assert report.total_nodes == 2
    assert any("Error Trigger" in f for f in report.red_flags)
    assert any("Google" in f for f in report.red_flags)
    assert any("PII field hint" in f for f in report.red_flags)
    assert "graph TD" in report.mermaid_graph
