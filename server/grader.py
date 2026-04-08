
"""
Grader for the Hydropower Dam Management environment.

Runs after each episode ends. Reads final stats from state()
and scores 0.0 to 1.0 per task based on task-specific criteria.

Task 1 — Dry Season Stability
    0.5 * eco_compliance_rate         (fraction of steps with no eco violation)
    0.5 * reservoir_stability_rate    (fraction of steps reservoir stayed above 30%)

Task 2 — Monsoon Flood Control
    0.6 * flood_free_score            (1.0 if zero floods, decays by 0.1 per flood event)
    0.4 * power_score                 (MWh generated / target MWh, capped at 1.0)

Task 3 — Price-Aware Dispatch
    0.5 * revenue_score               (actual revenue / theoretical max, capped at 1.0)
    0.3 * flood_compliance_score      (fraction of steps flood-free, full marks only if > 95%)
    0.2 * eco_compliance_score        (fraction of steps eco-compliant, full marks only if > 95%)
"""

from dataclasses import dataclass
from typing import Dict
from configs import TASKS


@dataclass
class GradeResult:
    task_id: str
    score: float                          # final 0.0 to 1.0
    breakdown: Dict[str, float]           # component scores for transparency
    passed: bool                          # score >= 0.5 considered a pass
    summary: str                          # human-readable verdict


def grade(task_id: str, episode_stats: dict) -> GradeResult:
    """
    Grade a completed episode.

    Args:
        task_id: one of "task_1", "task_2", "task_3"
        episode_stats: dict with keys:
            - total_steps         (int)
            - flood_count         (int)
            - eco_violation_count (int)
            - spill_count         (int)
            - total_power_mwh     (float)
            - total_revenue_inr   (float)
            - steps_below_reservoir_threshold (int)  — tracked by env

    Returns:
        GradeResult with score, breakdown, and summary
    """

    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    task = TASKS[task_id]
    grader_cfg = task["grader"]
    total_steps = max(episode_stats["total_steps"], 1)  # avoid div by zero

    # ------------------------------------------------------------------
    # Task 1 — Dry Season Stability
    # ------------------------------------------------------------------
    if task_id == "task_1":
        eco_violations = episode_stats["eco_violation_count"]
        steps_below_reservoir = episode_stats["steps_below_reservoir_threshold"]

        eco_compliance_rate = 1.0 - (eco_violations / total_steps)
        reservoir_stability_rate = 1.0 - (steps_below_reservoir / total_steps)

        # Clamp to [0, 1]
        eco_compliance_rate = max(0.0, min(1.0, eco_compliance_rate))
        reservoir_stability_rate = max(0.0, min(1.0, reservoir_stability_rate))

        score = (
            grader_cfg["eco_compliance_weight"] * eco_compliance_rate +
            grader_cfg["reservoir_stability_weight"] * reservoir_stability_rate
        )

        breakdown = {
            "eco_compliance_rate": round(eco_compliance_rate, 4),
            "reservoir_stability_rate": round(reservoir_stability_rate, 4),
        }

        summary = (
            f"Eco compliance: {eco_compliance_rate*100:.1f}% of steps | "
            f"Reservoir stable: {reservoir_stability_rate*100:.1f}% of steps | "
            f"Score: {score:.3f}"
        )

    # ------------------------------------------------------------------
    # Task 2 — Monsoon Flood Control
    # ------------------------------------------------------------------
    elif task_id == "task_2":
        flood_count = episode_stats["flood_count"]
        total_power_mwh = episode_stats["total_power_mwh"]
        power_target = grader_cfg["power_target_mwh"]

        # Flood score: starts at 1.0, loses 0.1 per flood event, floor at 0.0
        flood_free_score = max(0.0, 1.0 - (flood_count * 0.1))

        # Power score: fraction of target achieved, capped at 1.0
        power_score = min(1.0, total_power_mwh / power_target)

        score = (
            grader_cfg["flood_free_weight"] * flood_free_score +
            grader_cfg["power_weight"] * power_score
        )

        breakdown = {
            "flood_count": flood_count,
            "flood_free_score": round(flood_free_score, 4),
            "power_generated_mwh": round(total_power_mwh, 2),
            "power_target_mwh": power_target,
            "power_score": round(power_score, 4),
        }

        summary = (
            f"Floods: {flood_count} (score: {flood_free_score:.2f}) | "
            f"Power: {total_power_mwh:.0f}/{power_target} MWh (score: {power_score:.2f}) | "
            f"Score: {score:.3f}"
        )

    # ------------------------------------------------------------------
    # Task 3 — Price-Aware Dispatch
    # ------------------------------------------------------------------
    elif task_id == "task_3":
        total_revenue = episode_stats["total_revenue_inr"]
        flood_count = episode_stats["flood_count"]
        eco_violations = episode_stats["eco_violation_count"]
        theoretical_max = grader_cfg["theoretical_max_revenue"]

        flood_compliance_threshold = grader_cfg["flood_compliance_threshold"]
        eco_compliance_threshold = grader_cfg["eco_compliance_threshold"]

        # Revenue score: fraction of theoretical max, capped at 1.0
        revenue_score = min(1.0, total_revenue / theoretical_max)

        # Flood compliance: fraction of steps flood-free
        flood_compliance_rate = 1.0 - (flood_count / total_steps)
        flood_compliance_rate = max(0.0, min(1.0, flood_compliance_rate))
        # Only full marks if above threshold, otherwise scale proportionally
        flood_compliance_score = (
            flood_compliance_rate / flood_compliance_threshold
            if flood_compliance_rate < flood_compliance_threshold
            else 1.0
        )

        # Eco compliance: same logic
        eco_compliance_rate = 1.0 - (eco_violations / total_steps)
        eco_compliance_rate = max(0.0, min(1.0, eco_compliance_rate))
        eco_compliance_score = (
            eco_compliance_rate / eco_compliance_threshold
            if eco_compliance_rate < eco_compliance_threshold
            else 1.0
        )

        score = (
            grader_cfg["revenue_weight"] * revenue_score +
            grader_cfg["flood_compliance_weight"] * flood_compliance_score +
            grader_cfg["eco_compliance_weight"] * eco_compliance_score
        )

        breakdown = {
            "total_revenue_inr": round(total_revenue, 2),
            "theoretical_max_inr": theoretical_max,
            "revenue_score": round(revenue_score, 4),
            "flood_compliance_rate": round(flood_compliance_rate, 4),
            "flood_compliance_score": round(flood_compliance_score, 4),
            "eco_compliance_rate": round(eco_compliance_rate, 4),
            "eco_compliance_score": round(eco_compliance_score, 4),
        }

        summary = (
            f"Revenue: {total_revenue:.0f}/{theoretical_max} INR (score: {revenue_score:.2f}) | "
            f"Flood compliance: {flood_compliance_rate*100:.1f}% | "
            f"Eco compliance: {eco_compliance_rate*100:.1f}% | "
            f"Score: {score:.3f}"
        )

    score = round(max(0.0, min(1.0, score)), 4)

    return GradeResult(
        task_id=task_id,
        score=score,
        breakdown=breakdown,
        passed=score >= 0.5,
        summary=summary,
    )


def grade_all(results_by_task: dict) -> dict:
    """
    Grade all three tasks at once.

    Args:
        results_by_task: { "task_1": episode_stats, "task_2": ..., "task_3": ... }

    Returns:
        {
            "task_1": GradeResult,
            "task_2": GradeResult,
            "task_3": GradeResult,
            "overall": float   # mean score across all tasks
        }
    """
    grades = {}
    for task_id, stats in results_by_task.items():
        grades[task_id] = grade(task_id, stats)

    overall = sum(g.score for g in grades.values()) / len(grades)
    grades["overall"] = round(overall, 4)

    return grades
