# edge-inference-memory-profiler

Estimate activation memory and weight storage from ONNX models to support edge deployment feasibility analysis.

This tool provides a lightweight, deployment-oriented memory estimate for ONNX models targeting resource-constrained systems.

## What it does

- Estimates **weight storage** (Flash/ROM proxy) from ONNX initializers
- Estimates **activation tensor memory** (RAM proxy) from inferred tensor shapes
- Reports **naive peak** activation memory (no reuse)
- Reports **reuse peak** activation memory using a greedy lifetime-based buffer reuse simulation
- Includes an **input baseline** in reuse peak reporting for correctness

## Quick Example

```bash
eimp examples/tiny.onnx