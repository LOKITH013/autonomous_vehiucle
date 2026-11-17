"""
M12 — Decision engine (Sec 16).

Converts (risk, ttc, traffic_light, in_path) into one of the recommendation
labels from Sec 16. Every output carries the explanation string(s) that
produced it — these are RESEARCH PROTOTYPE RECOMMENDATIONS, never
vehicle-control commands (Sec 16, Sec 38).
"""


def decide(cfg, *, risk, ttc, traffic_light=None, obstacle_in_path=False,
           risk_reasons=None):
    dcfg = cfg["decision"]
    reasons = list(risk_reasons or [])

    if traffic_light == "RED":
        return "STOP", reasons + ["Traffic light RED"]

    if obstacle_in_path:
        return "AVOID OBSTACLE", reasons + ["Obstacle occupies current path"]

    if ttc is not None and ttc != float("inf"):
        if ttc < dcfg["brake_ttc_s"]:
            return "BRAKE", reasons + [f"TTC {ttc:.1f}s below brake threshold "
                                        f"{dcfg['brake_ttc_s']}s"]
        if ttc < dcfg["slow_down_ttc_s"]:
            return "SLOW DOWN", reasons + [f"TTC {ttc:.1f}s below slow-down "
                                            f"threshold {dcfg['slow_down_ttc_s']}s"]

    if risk == "HIGH":
        return "SLOW DOWN", reasons + ["Overall risk classified HIGH"]
    if risk == "MEDIUM":
        return "MONITOR", reasons + ["Overall risk classified MEDIUM"]

    return "MAINTAIN SPEED", reasons + ["No elevated risk detected"]
