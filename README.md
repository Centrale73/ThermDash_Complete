# ThermDash Complete

**ThermDash Complete** is a local-first AI efficiency and workflow assessment toolkit built by Central 73. It helps small and medium-sized businesses uncover duplicated AI tool spend, workflow rework, cloud dependencies, and—when compute evidence is available—per-request energy and heat footprints.

## Business Outcomes First

1. **Tool Inventory & Visibility**: Scans n8n automations and AI usage to produce a structured CSV/JSON inventory of active models, vendors, and cloud endpoints—supporting the first discovery step for Quebec Law 25 governance (not a legal compliance determination).
2. **Rework & Reliability Diagnostics**: Flags missing error triggers, hardcoded memory keys, unpinned model versions, and multi-turn rework penalties.
3. **Evidence-Backed Energy & Thermodynamics**: Estimates per-request and per-workflow-run energy (Joules) with IQR confidence intervals, distinguishing standard models from reasoning/test-time-scaling models.

## Quick Start

```bash
python cli.py calc-energy --model o3-mini --in 1200 --out 500 --rework-turns 2
python cli.py audit-n8n path/to/workflow.json --inventory-out inventory.csv --mermaid-out graph.mmd
pytest
```

## Architecture

- `energy.json` — Model energy constants with prefill/decode weights, reasoning multipliers, IQR bounds, and quantization flags.
- `token_dashboard/thermodynamics.py` — Core `E_in`, `E_out`, `W_useful`, `eta`, rework discount, and IQR interval calculations.
- `token_dashboard/energy_table.py` — Cached model lookup with snapshot stripping and fuzzy fallback.
- `token_dashboard/n8n_audit.py` — Modular n8n workflow auditor (synthesized from `ThermDash-AutoAudit`) with Mermaid graph and per-run energy estimation.
- `token_dashboard/inventory.py` — CSV/JSON AI tool inventory exporter.
- `cli.py` — Unified command-line interface.
