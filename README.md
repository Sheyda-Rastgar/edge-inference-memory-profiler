# edge-inference-memory-profiler

Estimate activation memory and weight storage from ONNX models to support edge deployment feasibility analysis.

## Motivation

When deploying neural networks on resource-constrained systems, RAM is often the real bottleneck — not model size.

While many frameworks provide runtime profiling, memory feasibility is usually evaluated late in the pipeline (after conversion, integration, or hardware testing). Redesigning a model at that stage can be costly.

This project explores a lightweight, model-level estimation approach:

> How much peak activation memory will this ONNX model require during inference — before deployment?

---

## What This Tool Provides

- Estimates **weight storage** (Flash/ROM proxy) from ONNX initializers  
- Estimates **activation tensor memory** (RAM proxy) using inferred tensor shapes  
- Reports **naive peak** activation memory (simple summation)  
- Reports **reuse-aware peak** using tensor lifetime analysis and greedy buffer reuse  
- Includes **input baseline correction** for more realistic peak estimation  

The reuse-aware estimate provides a tighter upper bound compared to naive summation.

---

## What It Is Not

- Not a runtime profiler  
- Not tied to a specific framework (TFLite / ONNX Runtime / CMSIS-NN)  
- Does not simulate kernel-level temporary workspace memory  

This tool focuses on **graph-level tensor memory estimation**.

---

## Quick Example

```bash
eimp examples/tiny.onnx
```

## Example output:
Estimated weight storage: ...
Naive peak activations (no reuse): ...
Reuse peak activations (greedy): ...
Estimated reduction: ...

## Design Philosophy
	•	Lightweight
	•	Framework-agnostic (ONNX-based)
	•	Deployment-oriented
	•	Focused on memory-aware model reasoning

---
## Future Development

Planned improvements include:
	•	Structured JSON output for automated benchmarking
	•	More accurate lifetime modeling (including input/output lifetimes)
	•	Alternative allocator strategies (first-fit, best-fit simulation)
	•	Optional fragmentation-aware peak estimation
	•	Improved support for partially dynamic shapes
	•	Operator-level breakdown for activation bottleneck identification
