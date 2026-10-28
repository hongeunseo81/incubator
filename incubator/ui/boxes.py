def normalize_box(x0, y0, x1, y1, w, h, min_size=3) -> tuple[int, int, int, int] | None:
    """드래그 방향과 무관하게 정렬하고 이미지 안으로 자른다. 너무 작으면 None."""
    xa, xb = sorted((round(x0), round(x1)))
    ya, yb = sorted((round(y0), round(y1)))
    xa, xb = max(0, xa), min(w, xb)
    ya, yb = max(0, ya), min(h, yb)
    if xb - xa < min_size or yb - ya < min_size:
        return None
    return xa, ya, xb, yb
