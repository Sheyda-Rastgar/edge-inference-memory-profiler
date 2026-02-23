# edge-inference-memory-profiler

Estimate activation memory and weight storage from ONNX models to support edge deployment feasibility analysis.

This tool provides a lightweight, deployment-oriented memory estimate for ONNX models targeting resource-constrained systems.

## What it does

- Estimates weight storage (Flash proxy)
- Estimates activation tensor memory (RAM proxy)
- Reports naive peak activation memory (no buffer reuse simulation yet)

## Quick Example

```bash
eimp examples/tiny.onnx