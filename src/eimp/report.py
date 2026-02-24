def format_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TB"


def format_percent(x: float) -> str:
    return f"{x:.1f}%"


def print_report(weight_bytes: int, naive_peak: int, reuse_peak: int, meta: dict | None = None) -> None:
    print("\n=== Edge Inference Memory Report ===\n")
    print(f"Estimated weight storage: {format_bytes(weight_bytes)}")
    print(f"Naive peak activations (no reuse): {format_bytes(naive_peak)}")
    print(f"Reuse peak activations (greedy):   {format_bytes(reuse_peak)}")

    if meta is not None:
        input_b = int(meta.get("input_bytes", 0))
        raw_b = int(meta.get("reuse_peak_bytes_raw", 0))
        print(f"Input baseline included:            {format_bytes(input_b)}")
        print(f"Reuse peak (raw, no inputs):        {format_bytes(raw_b)}")

    if naive_peak > 0:
        reduction = (naive_peak - reuse_peak) / naive_peak * 100.0
        print(f"Estimated reduction:               {format_percent(reduction)}")
    print("")