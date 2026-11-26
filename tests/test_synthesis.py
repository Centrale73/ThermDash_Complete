"""Smoke tests for ThermDash Complete synthesis."""

from token_dashboard.energy_table import lookup
from token_dashboard.inventory import export_inventory_csv, inventory_from_n8n
from token_dashboard.n8n_audit import N8nWorkflowAuditor
from token_dashboard.thermodynamics import efficiency


def test_reasoning_multiplier_and_iqr():
    c = lookup("o3-mini")
    rep = efficiency(n_in=1000, n_out=500, c=c, q=0.8, rework_turns=2)
    assert c.is_reasoning is True
    assert rep.e_in_low < rep.e_in < rep.e_in_high
    assert rep.rework_turns == 2
    assert 0.0 <= rep.eta <= 1.0


def test_n8n_audit_and_inventory():
    sample_workflow = {
        "name": "Lead Triage",
        "nodes": [
            {
                "name": "OpenAI Chat",
                "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
                "parameters": {"model": "gpt-4o", "prompt": "Summarize client email"},
            },
            {
                "name": "Local Ollama",
                "type": "@n8n/n8n-nodes-langchain.lmChatOllama",
                "parameters": {"model": "llama-3.1-70b"},
            },
        ],
        "connections": {},
    }
    auditor = N8nWorkflowAuditor(sample_workflow)
    report = auditor.audit(assumed_in_tokens=1000, assumed_out_tokens=400)
    assert report.total_nodes == 2
    assert report.estimated_run_joules > 0.0
    assert any("Error Trigger" in f for f in report.red_flags)
    assert any("email" in f for f in report.red_flags)

    items = inventory_from_n8n(report)
    csv_text = export_inventory_csv(items)
    assert "OpenAI Chat" in csv_text
    assert "gpt-4o" in csv_text
