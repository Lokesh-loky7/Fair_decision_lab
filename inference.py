# inference.py - runs the agent against all tasks and logs results.
# Must be at project ROOT. Reads HF_TOKEN only.

import os
import sys
import json

from openai import OpenAI

# Fix imports for when running from HF Space
try:
    from fair_decision_lab.client import FairDecisionEnv
    from fair_decision_lab.models import FairAction
except ModuleNotFoundError:
    # Running from HF Space where files are at root
    from client import FairDecisionEnv
    from models import FairAction

# ---------------------------------------------------------------------------
# Config - all from environment, exit clearly if missing
# ---------------------------------------------------------------------------

def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required environment variable {name!r} is not set.", flush=True)
        sys.exit(1)
    return val

HF_TOKEN     = _require("HF_TOKEN")
API_BASE_URL = _require("API_BASE_URL")
MODEL_NAME   = _require("MODEL_NAME")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = (
    "You are an AI bias auditor. You will receive a dataset summary and must analyse it for bias. "
    "Respond ONLY with a JSON object containing exactly these keys:\n"
    '  "identified_bias": true or false\n'
    '  "group_gap_estimate": a number in percentage points, or null\n'
    '  "analysis": your written reasoning (at least 60 words)\n'
    '  "recommended_action": MUST be one of "reweight", "remove_feature", "oversample", or null\n'
    "Do not include any text outside the JSON object. "
    "For recommended_action, use EXACTLY one of these words: reweight, remove_feature, oversample."
)

TASKS    = ["easy", "medium", "hard"]
EPISODES = 3


def _call_llm(messages: list) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=400, timeout=60,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"[LLM_ERROR] {e}", flush=True)
        return {}


def _build_action(parsed: dict) -> FairAction:
    return FairAction(
        analysis=str(parsed.get("analysis", "[no analysis provided]")),
        identified_bias=bool(parsed.get("identified_bias", False)),
        group_gap_estimate=float(parsed["group_gap_estimate"]) if parsed.get("group_gap_estimate") is not None else None,
        recommended_action=parsed.get("recommended_action"),
    )


def run_episode(task_id: str, episode_num: int):
    # Log START in required format
    print(f"[START] task={task_id} env=fair-decision-lab model={MODEL_NAME}", flush=True)

    rewards_list = []
    steps = 0
    success = False
    
    try:
        with FairDecisionEnv(base_url=ENV_URL).sync() as env:
            reset_result = env.reset(task_id=task_id)
            obs = reset_result.observation

            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": f"Dataset summary:\n{obs.dataset_summary}"},
            ]

            done = False

            while not done:
                parsed = _call_llm(messages)
                action = _build_action(parsed)

                try:
                    result   = env.step(action)
                    next_obs = result.observation
                    reward   = result.reward or 0.0
                    done     = result.done
                    error    = None
                except Exception as e:
                    error = str(e)
                    print(f"[STEP] step={steps+1} action={action.analysis[:50]} reward=0.00 done=true error={error}", flush=True)
                    break

                steps += 1
                rewards_list.append(reward)
                
                # Log STEP in required format
                action_str = action.analysis[:50].replace('\n', ' ')
                done_str = str(done).lower()
                print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error=null", flush=True)

                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                if not done:
                    messages.append({"role": "user", "content": (
                        f"Feedback: {next_obs.feedback}\n"
                        f"Dataset summary:\n{next_obs.dataset_summary}"
                    )})

            # Calculate score (normalized to [0, 1])
            total_reward = sum(rewards_list)
            # For easy: max ~0.95, medium: ~0.70, hard: ~1.20 (3 turns * 0.40)
            max_possible = 0.95 if task_id == "easy" else (0.70 if task_id == "medium" else 1.20)
            score = min(total_reward / max_possible, 1.0) if max_possible > 0 else 0.0
            success = score >= 0.5  # Success if score >= 50%

    except Exception as e:
        print(f"[DEBUG] episode failed: {e}", flush=True)
        score = 0.0
        success = False

    # Log END in required format
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    success_str = str(success).lower()
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task_id in TASKS:
        for episode_num in range(1, EPISODES + 1):
            run_episode(task_id, episode_num)