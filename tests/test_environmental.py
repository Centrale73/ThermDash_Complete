"""Tests for EcoLogits & CodeCarbon integration and separation from financial cost."""

from token_dashboard.environmental import LocalCodeCarbonTracker, estimate_cloud_impact
from token_dashboard.n8n_audit import N8nWorkflowAuditor


def test_unsupported_model_returns_not_estimated_not_zero():
    est = estimate_cloud_impact("openai", "nonexistent-model-xyz", output_tokens=300)
    assert est.status == "not_estimated"
    assert est.energy_kwh is None
    assert len(est.warnings) >= 1


def test_local_codecarbon_context_manager_returns_structured_result():
    with LocalCodeCarbonTracker(project_name="test_eval") as tracker:
        _ = sum(range(1000))
    assert tracker.result is not None
    assert tracker.result.status in ("measured", "not_measured")
    assert "CPU" in tracker.result.boundary_note


def test_n8n_audit_separates_cost_and_environmental_estimates():
    wf = {
        "name": "Email Digest",
        "nodes": [
            {
                "name": "Summarize",
                "type": "@n8n/n8n-nodes-langchain.openAi",
                "parameters": {"modelId": {"value": "gpt-4o-mini"}},
            }
        ],
        "connections": {},
    }
    report = N8nWorkflowAuditor(wf).audit()
    assert report.estimated_run_cost_usd is not None
    assert report.estimated_run_cost_usd > 0.0
    assert len(report.environmental_estimates) == 1
    assert report.environmental_estimates[0]["status"] in ("estimated", "not_estimated")
