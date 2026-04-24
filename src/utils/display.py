

def print_optimised_corners_values(corners):
    rows = []
    for corner in corners:
        rows.append({
            "corner": list(corner[0]) if not isinstance(corner[0], dict) else corner[0],
            "value": float(corner[2]),
        })
    rows_sorted = sorted(rows, key=lambda r: r["value"])
    print("[Corners]:")
    for r in rows_sorted:
        print(f"Corner: {r["corner"]} => Estimated KSD: {r["value"]:.3f}")