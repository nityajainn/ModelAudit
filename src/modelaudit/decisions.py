"""Explicit review findings and scalar policy gates, without executable paths."""

import math


def evaluate_gates(results, gates):
    decisions = []
    for gate in gates:
        value = results
        reason = None
        for part in gate["path"]:
            if isinstance(value, dict) and isinstance(part, str) and part in value:
                value = value[part]
            elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                value = value[part]
            else:
                reason = "Configured result path is absent or has the wrong container type."
                break
        if reason is None and (type(value) not in (int, float) or not math.isfinite(value)):
            reason = "Result is undefined or is not a finite numeric measurement."
        if reason:
            status, value = "unavailable", None
        else:
            status = "pass" if gate.get("minimum", -math.inf) <= value <= gate.get("maximum", math.inf) else "fail"
        decisions.append({**gate, "value": value, "status": status, "reason": reason})
    return decisions


def summarize_results(results, gates):
    """Keep warning locations distinct; do not treat missing evidence as a pass."""
    review_items, failures = [], []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "warnings" and isinstance(child, list):
                    review_items.extend({"path": path + [key, index], "detail": item}
                                        for index, item in enumerate(child))
                elif key == "findings" and isinstance(child, list):
                    for index, item in enumerate(child):
                        entry = {"path": path + [key, index], "detail": item}
                        if item.get("status") == "fail":
                            failures.append(entry)
                        elif item.get("status") in ("warn", "skipped"):
                            review_items.append(entry)
                elif key == "undefined_reasons" and child:
                    review_items.append({"path": path + [key], "detail": child})
                elif key == "undefined_reason" and child:
                    review_items.append({"path": path + [key], "detail": child})
                elif key not in ("context", "inputs", "limitations"):
                    visit(child, path + [key])
            if value.get("status") == "skipped":
                review_items.append({"path": path, "detail": value.get("reason", "Measurement skipped.")})
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + [index])

    visit(results, [])
    gate_results = evaluate_gates(results, gates)
    failed = bool(failures) or any(gate["status"] == "fail" for gate in gate_results)
    review = bool(review_items) or any(gate["status"] == "unavailable" for gate in gate_results)
    return {"status": "fail" if failed else "review" if review else "pass",
            "exit_code": 1 if failed or review else 0,
            "review_items": review_items, "data_failures": failures, "gates": gate_results,
            "meaning": "Outcome of configured offline checks only; not deployment approval."}
