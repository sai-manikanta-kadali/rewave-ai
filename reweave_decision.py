"""Explainable human-in-the-loop routing engine for ReWeave AI.

Strictly separates:
1. Model detections (YOLO suggestions only)
2. OCR evidence (explicit label data; UNKNOWN if unreadable)
3. Worker inputs (human-observed condition & damage)
4. Rules-based recommendation (with conflict checks)
5. Human confirmation (always required before final routing)
"""

from __future__ import annotations

from typing import Any

PATHWAY_REUSE = "REUSE"
PATHWAY_REPAIR = "REPAIR"
PATHWAY_RECOVERY = "MATERIAL TESTING / RECOVERY ASSESSMENT"
PATHWAY_MANUAL = "MANUAL INSPECTION"

_DAMAGE_IGNORES = {"No visible damage", "Other / unsure"}


def evaluate_garment(
    worker_condition: str,
    observed_damage: list[str] | None = None,
    yolo_detections: list[dict[str, Any]] | None = None,
    ocr_result: dict[str, Any] | None = None,
    conflict_conf_threshold: float = 0.45,
) -> dict[str, Any]:
    """Generate an explainable sorting recommendation from all available evidence."""
    damage_list = list(observed_damage or [])
    detections = list(yolo_detections or [])
    ocr = ocr_result or {}

    composition = ocr.get("composition") or "UNKNOWN"
    composition_evidence = ocr.get("composition_evidence") or (
        "Care-label OCR" if composition != "UNKNOWN" else "None (composition UNKNOWN)"
    )
    care_instructions = list(ocr.get("care_instructions") or [])

    high_conf_defects = [
        d for d in detections if float(d.get("confidence", 0.0)) >= conflict_conf_threshold
    ]
    reported_defects = [d for d in damage_list if d not in _DAMAGE_IGNORES]

    reasons: list[str] = []
    conflict_flag = False

    if worker_condition == "Beyond repair":
        pathway = PATHWAY_RECOVERY
        reasons.append("Direct reuse/repair not established (worker assessed garment as beyond repair).")
        if detections:
            detected_names = ", ".join(sorted({d["class_name"] for d in detections}))
            reasons.append(f"YOLO model also flagged visible defect(s): {detected_names}.")
        if composition == "UNKNOWN":
            reasons.append("Fibre composition is UNKNOWN — manual material verification required before recovery/recycling.")
        else:
            reasons.append(f"Care-label OCR indicates {composition} — verify physical tag before material-specific recovery.")

    elif worker_condition == "Minor repair needed":
        pathway = PATHWAY_REPAIR
        reasons.append("Minor repairable damage reported by worker — prioritise REPAIR before recycling.")
        if reported_defects:
            reasons.append(f"Worker-selected damage: {', '.join(reported_defects)}.")
        if detections:
            detected_names = ", ".join(sorted({d["class_name"] for d in detections}))
            reasons.append(f"YOLO vision support flagged: {detected_names} (verify severity before repair).")

    elif worker_condition == "Ready to wear":
        if high_conf_defects or reported_defects:
            pathway = PATHWAY_MANUAL
            conflict_flag = True
            reasons.append(
                "Human–AI / input conflict: Worker selected 'Ready to wear', but potential damage was flagged."
            )
            if high_conf_defects:
                flagged = ", ".join(f"{d['class_name']} ({d['confidence']:.0%})" for d in high_conf_defects)
                reasons.append(f"YOLO flagged high-confidence defect(s): {flagged}. Re-inspect before confirming REUSE.")
            if reported_defects:
                reasons.append(f"Observed damage checkboxes selected: {', '.join(reported_defects)}.")
        else:
            pathway = PATHWAY_REUSE
            reasons.append("Worker assessed garment as ready to wear and no high-confidence defects were flagged.")
            if detections:
                reasons.append(
                    "Note: Low-confidence model suggestion(s) were below conflict threshold — quick visual check advised."
                )
            reasons.append("Confirm hygiene, odour, and fit before resale or donation.")

    else:
        pathway = PATHWAY_MANUAL
        reasons.append("Worker marked condition as uncertain — manual inspection required (never auto-routed by YOLO alone).")
        if detections:
            flagged = ", ".join(f"{d['class_name']} ({d['confidence']:.0%})" for d in detections)
            reasons.append(f"YOLO suggestions to inspect closely: {flagged}.")
        else:
            reasons.append("No YOLO defects detected above threshold (absence of detection is not proof garment is defect-free).")

    return {
        "recommended_pathway": pathway,
        "reasons": reasons,
        "rationale": " ".join(reasons),
        "composition": composition,
        "composition_evidence": composition_evidence,
        "care_instructions": care_instructions,
        "yolo_detection_count": len(detections),
        "yolo_classes": sorted({d["class_name"] for d in detections}),
        "conflict_flag": conflict_flag,
        "human_verification_required": True,
    }
