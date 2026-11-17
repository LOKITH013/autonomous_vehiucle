"""
M11 — Risk assessment (Sec 15).

Deliberately rule-based (per spec's own rationale: explainability, easy
debugging, easy viva defense, clear input->output traceability). Thresholds
come from config.yaml and must be treated as tunable project parameters,
not universal safety constants (Sec 15).

Lane position and traffic-light state are accepted as optional inputs so
this engine slots in cleanly once M5 (lane detection) and M6 (traffic
sign/light) are implemented; until then, pass None and this engine simply
skips those factors (documented, not silently faked).
"""

RISK_LOW, RISK_MEDIUM, RISK_HIGH = "LOW", "MEDIUM", "HIGH"


def assess_risk(cfg, *, ttc, distance_m, cls_name,
                 lane_position=None, traffic_light=None, in_path=True):
    """
    Returns (risk_level: str, reasons: list[str]) — reasons make the
    decision explainable (Sec 15, Sec 16).
    """
    reasons = []
    risk = RISK_LOW

    rcfg = cfg["risk"]

    if traffic_light == "RED":
        return RISK_HIGH, ["Traffic light is RED"]

    if ttc is not None and ttc != float("inf") and in_path:
        if ttc < rcfg["ttc_high_risk_s"]:
            risk = RISK_HIGH
            reasons.append(f"TTC {ttc:.1f}s < high-risk threshold "
                            f"{rcfg['ttc_high_risk_s']}s")
        elif ttc < rcfg["ttc_medium_risk_s"]:
            risk = _max_risk(risk, RISK_MEDIUM)
            reasons.append(f"TTC {ttc:.1f}s < medium-risk threshold "
                            f"{rcfg['ttc_medium_risk_s']}s")

    if distance_m is not None and in_path:
        if distance_m < rcfg["close_distance_high_m"]:
            risk = _max_risk(risk, RISK_HIGH)
            reasons.append(f"Distance {distance_m:.1f}m < "
                            f"{rcfg['close_distance_high_m']}m")
        elif distance_m < rcfg["close_distance_medium_m"]:
            risk = _max_risk(risk, RISK_MEDIUM)
            reasons.append(f"Distance {distance_m:.1f}m < "
                            f"{rcfg['close_distance_medium_m']}m")

    if cls_name in ("person", "bicycle", "motorcycle") and risk == RISK_MEDIUM:
        # vulnerable road users get bumped up one level at medium risk
        risk = RISK_HIGH
        reasons.append(f"Vulnerable road user ({cls_name}) at medium-risk range")

    if lane_position == "OFF_LANE":
        risk = _max_risk(risk, RISK_MEDIUM)
        reasons.append("Ego vehicle position estimated off current lane")

    if not reasons:
        reasons.append("No elevated-risk factors detected")

    return risk, reasons


def _max_risk(a, b):
    order = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}
    return a if order[a] >= order[b] else b
