"""JSON/YAML-safe metric values, without clipping mathematically perfect PSNR."""
import math


def finite_metric_values(value):
    if isinstance(value, dict):
        return {key: finite_metric_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_metric_values(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or value == -math.inf:
            raise ValueError("Unexpected non-finite metric")
        if value == math.inf:
            return None
    return value


def psnr_report_summary(report, count):
    """Validate v3 perfect markers; legacy reports must contain finite values."""
    values = report.get("psnr_2d_projs")
    if not isinstance(values, list) or len(values) != count or count < 1:
        raise ValueError("PSNR per-view metrics have the wrong length")
    modern = report.get("projection_metric_version") == "global_reference_v3"
    statuses = report.get("psnr_2d_projs_status") if modern else ["finite"] * count
    if not isinstance(statuses, list) or len(statuses) != count:
        raise ValueError("PSNR per-view statuses have the wrong length")
    finite = []
    perfect = 0
    for value, status in zip(values, statuses):
        if modern and status == "perfect" and value is None:
            perfect += 1
        elif status == "finite" and value is not None and math.isfinite(float(value)):
            finite.append(float(value))
        else:
            raise ValueError("Invalid PSNR value/status")
    overall = report.get("psnr_2d")
    if modern and perfect == count:
        if overall is not None or report.get("psnr_2d_status") != "perfect":
            raise ValueError("All-perfect PSNR must have a null overall score and perfect status")
    elif overall is None or not math.isfinite(float(overall)) or (modern and report.get("psnr_2d_status") != "finite"):
        raise ValueError("Invalid overall PSNR value/status")
    if modern and report.get("perfect_view_count") != perfect:
        raise ValueError("Perfect-view count does not match PSNR statuses")
    return {"overall": None if overall is None else float(overall),
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "perfect_view_count": perfect}
