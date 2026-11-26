# Known Limitations

1. **Estimated vs. Measured Energy**: Entries in `energy.json` with `"measured": false` are literature-derived estimates. Real-world inference energy varies substantially with batch size, GPU utilization, quantization, and serving engine (e.g., vLLM/TensorRT-LLM). Always review the IQR confidence bounds (`e_in_low`, `e_in_high`).
2. **Not Legal or Compliance Advice**: Regulatory hints (such as Quebec Law 25 Art. 17 or Art. 12 review flags) are engineering discovery indicators only. They do not constitute legal advice or a compliance certification.
3. **Static n8n Token Assumptions**: When auditing an n8n workflow JSON without runtime execution logs, per-run energy estimates rely on configurable assumed input/output token counts (`--in`, `--out`).
