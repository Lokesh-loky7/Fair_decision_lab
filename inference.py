# inference.py - runs the agent against all tasks and logs results.
# Must be at project ROOT. Reads HF_TOKEN only.

import os
import sys
import json

from openai import OpenAI

from fair_decision_lab.client import FairDecisionEnv
from fair_decision_lab.models import FairAction

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
    print("[START]", flush=True)
    print(f"task_id: {task_id}", flush=True)
    print(f"episode: {episode_num}", flush=True)

    try:
        with FairDecisionEnv(base_url=ENV_URL).sync() as env:
            reset_result = env.reset(task_id=task_id)
            obs = reset_result.observation

            messages = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": f"Dataset summary:\n{obs.dataset_summary}"},
            ]

            total_reward = 0.0
            steps        = 0
            done         = False

            while not done:
                parsed = _call_llm(messages)
                action = _build_action(parsed)

                try:
                    result   = env.step(action)
                    next_obs = result.observation
                    reward   = result.reward or 0.0
                    done     = result.done
                except Exception as e:
                    print(f"[ERROR] step failed: {e}", flush=True)
                    break

                steps        += 1
                total_reward += reward

                print("[STEP]", flush=True)
                print(f"step: {steps}", flush=True)
                print(f"action: {action.analysis[:120]!r}", flush=True)
                print(f"observation: {next_obs.feedback!r}", flush=True)
                print(f"reward: {reward:.4f}", flush=True)
                print(f"done: {done}", flush=True)

                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                if not done:
                    messages.append({"role": "user", "content": (
                        f"Feedback: {next_obs.feedback}\n"
                        f"Dataset summary:\n{next_obs.dataset_summary}"
                    )})

    except Exception as e:
        print(f"[ERROR] episode failed: {e}", flush=True)
        print("[END]", flush=True)
        print("total_reward: 0.0000", flush=True)
        print("steps: 0", flush=True)
        print("")
        return

    print("[END]", flush=True)
    print(f"total_reward: {total_reward:.4f}", flush=True)
    print(f"steps: {steps}", flush=True)
    print("")  # REQUIRED - parser separator


if __name__ == "__main__":
    for task_id in TASKS:
        for episode_num in range(1, EPISODES + 1):
            run_episode(task_id, episode_num)