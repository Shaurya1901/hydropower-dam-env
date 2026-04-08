
"""
Baseline Inference Script for Hydropower Dam Management Environment.

Runs an LLM agent (via OpenAI API) against all 3 tasks and produces
a reproducible score on each. Reads API key from environment variable.

Usage:
    OPENAI_API_KEY=your_key python baseline.py

The LLM agent receives the current observation as text and decides
the release level each step. The grader scores the final episode.
"""

import os
import sys
import json
from openai import OpenAI

sys.path.insert(0, '/content/hydropower_dam_env')
sys.path.insert(0, '/content/hydropower_dam_env/server')

from hydropower_dam_env_environment import HydropowerDamEnvEnvironment
from models import HydropowerAction
from grader import grade, grade_all
from configs import TASKS

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"       # cheap and fast for baseline
MAX_STEPS_PER_TASK = 2160   # full 90-day season
SEED = 42                   # reproducibility

# ------------------------------------------------------------------
# System prompt — tells the LLM what it is and what to do
# ------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an AI agent operating a hydropower dam.

Each step you receive the current state of the dam as a JSON observation.
You must respond with a single integer — your chosen release level.

Release levels:
0 = 0%   of max turbine capacity (hold water)
1 = 17%  of max turbine capacity
2 = 33%  of max turbine capacity
3 = 50%  of max turbine capacity
4 = 67%  of max turbine capacity
5 = 83%  of max turbine capacity
6 = 100% of max turbine capacity (full release)

Rules:
- Never let the downstream flow exceed the flood threshold
- Always maintain minimum ecological flow (release at least level 1)
- Keep the reservoir above 30% where possible
- Release more when grid price is high to maximise revenue

Respond with ONLY a single integer between 0 and 6. Nothing else.
"""

# ------------------------------------------------------------------
# LLM agent
# ------------------------------------------------------------------

def get_llm_action(client: OpenAI, obs, task_description: str) -> int:
    """
    Ask the LLM to pick a release level given the current observation.
    Falls back to level 3 (50%) if the response is invalid.
    """
    obs_text = json.dumps({
        "reservoir_level": round(obs.reservoir_level, 3),
        "inflow_forecast_m3s": [round(x, 1) for x in obs.inflow_forecast],
        "grid_price_inr_kwh": round(obs.grid_price, 2),
        "downstream_demand_m3s": obs.downstream_demand,
        "step": obs.step,
        "flood_triggered_last_step": obs.flood_triggered,
        "eco_violation_last_step": obs.eco_violation,
        "task": task_description,
    }, indent=2)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            max_tokens=5,
            temperature=0.0,   # deterministic
        )
        raw = response.choices[0].message.content.strip()
        action = int(raw)
        if action < 0 or action > 6:
            raise ValueError(f"Out of range: {action}")
        return action

    except Exception as e:
        # Fallback to 50% release if LLM gives invalid response
        print(f"    [WARNING] LLM returned invalid action: {e}. Defaulting to level 3.")
        return 3


# ------------------------------------------------------------------
# Run one task
# ------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> dict:
    """Run a full episode on one task and return episode stats."""
    task = TASKS[task_id]
    print(f"
{'='*60}")
    print(f"  Task: {task_id} — {task['name']}")
    print(f"  Difficulty: {task['difficulty']}")
    print(f"  Objective: {task['objective']}")
    print(f"{'='*60}")

    env = HydropowerDamEnvEnvironment(task_id=task_id)
    obs = env.reset(seed=SEED)

    step = 0
    log_interval = 216   # log every ~9 days

    while not obs.done:
        action_int = get_llm_action(client, obs, task["description"])
        obs = env.step(HydropowerAction(release_level=action_int))
        step += 1

        if step % log_interval == 0:
            state = env.state
            print(
                f"  Day {state.season_day:>3} | "
                f"Reservoir: {obs.reservoir_level*100:.1f}% | "
                f"Price: {obs.grid_price:.2f} INR/kWh | "
                f"Action: {action_int} | "
                f"Reward: {obs.reward:.1f} | "
                f"Floods: {state.flood_count} | "
                f"Eco violations: {state.eco_violation_count}"
            )

    stats = env.episode_stats()
    print(f"
  Episode complete.")
    print(f"  Power generated : {stats['total_power_mwh']:.1f} MWh")
    print(f"  Revenue         : {stats['total_revenue_inr']:.0f} INR")
    print(f"  Flood events    : {stats['flood_count']}")
    print(f"  Eco violations  : {stats['eco_violation_count']}")
    print(f"  Spills          : {stats['spill_count']}")

    return stats


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Run: import os; os.environ['OPENAI_API_KEY'] = 'your_key'")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("
" + "="*60)
    print("  HYDROPOWER DAM — BASELINE INFERENCE")
    print("  Model:", MODEL)
    print("  Seed:", SEED)
    print("="*60)

    # Run all 3 tasks
    results = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        results[task_id] = run_task(client, task_id)

    # Grade all tasks
    grades = grade_all(results)

    # Print final scores
    print("
" + "="*60)
    print("  FINAL SCORES")
    print("="*60)
    for task_id in ["task_1", "task_2", "task_3"]:
        g = grades[task_id]
        status = "PASS" if g.passed else "FAIL"
        print(f"
  {task_id} [{status}]")
        print(f"  Score   : {g.score:.4f}")
        print(f"  Summary : {g.summary}")
        print(f"  Breakdown:")
        for k, v in g.breakdown.items():
            print(f"    {k}: {v}")

    print(f"
  OVERALL SCORE: {grades['overall']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
