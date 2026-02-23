def format_bytes(num: float) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TB"


def print_report(peak_bytes: int, activations, weight_bytes: int) -> None:
    print("\n=== Edge Inference Memory Report ===\n")
    print(f"Estimated weight storage: {format_bytes(weight_bytes)}")
    print(f"Naive peak activation memory (no reuse): {format_bytes(peak_bytes)}\n")

    print("Activation tensor estimates:")
    print("-" * 60)
    for name, size in activations:
        print(f"{name:40s} {format_bytes(size)}")