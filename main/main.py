import os
from google.genai import types
from functions import text_vectorize_score, ai_outputs
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import anthropic
from dotenv import load_dotenv
from enum import Enum
from bert_score import BERTScorer as TextBERT

load_dotenv(override=True)

model_text = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
bert_scorer = TextBERT(lang="en", model_type="distilbert-base-uncased")

print(f"All Models and Nessarcy imports loaded... \n")

COSINE_THRESHOLD = 0.82
MAX_REVIEW_B_RETRIES = 4

def build_pivot_prompt(review_a_text, failed_attempts):
    attempts_block = "\n\n".join(
        f" --- Review B Attempt {i + 1} (FAILED -- DO NOT REPEAT) ---\n{text}"
        for i, text in enumerate(failed_attempts)
    )

    return f"""
Review A is already recorded:

--- REVIEW A ---
{review_a_text}
--- END REVIEW A ---

Your prior Review B attempt(s) are below. Each one was found to collide
(too similar) with Review A or with each other:

{attempts_block}

You must now reason about a COMPLETELY DIFFERENT engineering taxonomy that
none of the above have meaningfully covered. Choose ONE of the following,
whichever has been touched least across all attempts above:

- security exploits (injection, auth bypass, unsafe deserialization)
- resource/scaling limits (memory growth, concurrency, throughput under load)
- failure modes (error handling, retries, partial-failure recovery)

Write a free-form analysis from within that taxonomy only. Do not restate
or rephrase anything from any attempt above, even indirectly.

Output only:

Review (pivoted):
[your analysis and recommendations]
    """

def run_review_b_with_retries(client, full_message, ai_type, review_a_text, cycle_num):
    failed_attempts = []
    attempt_log = []

    for attempt_num in range(1, MAX_REVIEW_B_RETRIES + 1):
        if attempt_num == 1:
            sys_prompt = PROMPTS[Phase.REVIEW_B]
            call_messages = full_message
        else:
            sys_prompt = build_pivot_prompt(review_a_text, failed_attempts)
            retry_cue = "Retry Review B with the pivoted taxonomy above"

            if ai_type == "gemini":
                call_messages = full_message + [{"role": "user", "parts": [{"text": retry_cue}]}]
            else:
                call_messages = full_message + [{"role": "user", "content": retry_cue}]

        response = ai_outputs(client, call_messages, ai_type, system_prompt=sys_prompt)

        score = text_vectorize_score(response, review_a_text, model_text, bert_scorer)
        cosine = score["cosine_similarity"]
        attempt_log.append({"attempt": attempt_num, "output": response, "cosine": cosine})
        print(f"\n--- cycle {cycle_num} / review_b attempt {attempt_num} (cosine={cosine:.3f}) ---\n{response}")


        if cosine <= COSINE_THRESHOLD:
            # only the winning attempt gets written into the real history
            if attempt_num > 1:
                if ai_type == "gemini":
                    full_message.append({"role": "user", "parts": [{"text": retry_cue}]})
                else:
                    full_message.append({"role": "user", "content": retry_cue})
            if ai_type == "gemini":
                full_message.append({"role": "model", "parts": [{"text": response}]})
            else:
                full_message.append({"role": "assistant", "content": response})
            return response, attempt_log

        failed_attempts.append(response)

    if MAX_REVIEW_B_RETRIES > 1:
        if ai_type == "gemini":
            full_message.append({"role": "user", "parts": [{"text": retry_cue}]})
        else:
            full_message.append({"role": "user", "content": retry_cue})
    if ai_type == "gemini":
        full_message.append({"role": "model", "parts": [{"text": response}]})
    else:
        full_message.append({"role": "assistant", "content": response})

    return response, attempt_log


REVIEW_A_SYS_PROMPT = """
You are an Architecture Reviewer examining a single piece of code.

Your ONLY lens: macro structure. You may discuss:
- module/function boundaries and responsibility splits
- data flow and state management design
- coupling between components
- scalability of the overall design under growth (more users, more data, more calls)
- whether the chosen approach/pattern fits the problem

Do NOT discuss (these belong to a separate reviewer):
- specific bugs, edge cases, or incorrect logic inside a function body
- security vulnerabilities
- syntax, naming, style, or formatting

Write a free-form analysis of the code from this lens. Think through what
is working, what is fragile, and what direction the next revision should
take. You are not producing a checklist -- write as if you are reasoning
out loud about the design and what should change and why.

Output only:

Review A:
[your analysis and recommendations]
"""

REVIEW_B_SYS_PROMPT = """
You are a QA Engineer examining a single piece of code. You have not
seen any other review of this code.

Your ONLY lens: micro execution correctness. You may discuss:
- specific bugs, off-by-one errors, incorrect conditionals
- untested edge cases (empty input, null, boundary values, concurrency)
- resource handling (leaks, unclosed connections, unbounded growth)
- security exploits (injection, unsafe deserialization, unchecked input)

Do NOT discuss (these belong to a separate reviewer):
- module boundaries, overall design pattern, or architecture-level
  restructuring
- style or naming preferences with no functional impact

Write a free-form analysis of the code from this lens. Trace through the
execution path, note where it breaks or behaves unexpectedly, and explain
what should be fixed and why.

Output only:

Review B:
[your analysis and recommendations]
"""

GENERATE_SYS_PROMPT = """
You are the Generator. Review A and Review B are already above you in
this conversation -- both are free-form analyses, not checklists.

Read both reviews, decide what they imply the code actually needs, and
produce a revised version that addresses the substance of both. You are
not required to address every sentence literally; use judgment about what
matters most and say briefly what you changed and why before the code.

Never repeat the previous draft verbatim.

Output only:

Generate:
[brief note on what you changed based on the reviews]
[revised code]
"""

class Phase(Enum):
    REVIEW_A = "review_a"
    REVIEW_B = "review_b"
    GENERATE = "generate"

ORDER = [Phase.REVIEW_A, Phase.REVIEW_B, Phase.GENERATE]
PROMPTS = {Phase.REVIEW_A: REVIEW_A_SYS_PROMPT,
           Phase.REVIEW_B: REVIEW_B_SYS_PROMPT,
           Phase.GENERATE: GENERATE_SYS_PROMPT}

NEXT_CUE = {Phase.REVIEW_A: "Proceed to Review B.",
            Phase.REVIEW_B: "Proceed to Generate.",
            Phase.GENERATE: "Proceed to the next cycle (Review A)."}

def run_normal_agent(client, message, ai_type, num_cycles):
    if ai_type == "claude" or ai_type == "chatgpt" or ai_type == "glm":
        full_message = [
            {"role": "user", "content": message}
        ]
    elif ai_type == "gemini":
        full_message = [
            {"role": "user", "parts": [{"text": message}]}
        ]
    else:
        raise ValueError("ai_type needs to be claude/chatgpt/gemini/glm")
    
    transcript = []
    remaining_cycles = num_cycles
    cycle_num = 1
    phase_idx = 0
    while (True):

        if phase_idx % 3 == 0:
            print(f"\n ===== Cycle {cycle_num} / {num_cycles} ===== ")

        phase = ORDER[phase_idx]

        alert = (f"\nSystem Alert: cycle {cycle_num}/{num_cycles}, "
                 f"phase = {phase.value}. {remaining_cycles} cycles remain.")
        
        if ai_type == "gemini":
            full_message[-1]["parts"][0]["text"] += alert
        else:
            full_message[-1]["content"] += alert

        response = ai_outputs(client, full_message, ai_type, system_prompt=PROMPTS[phase])
        transcript.append({"phase": phase.value, "cycle": cycle_num, "output": response})
        print(f"\n--- cycle {cycle_num} / {phase.value} ---\n{response}")

        if ai_type == "gemini":
            full_message.append({"role": "model", "parts": [{"text": response}]})
        else:
            full_message.append({"role": "assistant", "content": response})

        if phase == Phase.GENERATE:
            if "Final Answer:" in response:
                _, _, final = response.partition("Final Answer:")
                return {"final_answer": final.strip(), "transcript": transcript}
            remaining_cycles -= 1
            cycle_num += 1
            if remaining_cycles <= 0:
                return {"final_answer": None, "transcript": transcript, "note": "ran out of cycles without a Final Answer"}
            
        cue = NEXT_CUE[phase]
        if ai_type == "gemini":
            full_message.append({"role": "user", "parts": [{"text": cue}]})
        else:
            full_message.append({"role": "user", "content": cue})

        phase_idx = (phase_idx + 1) % len(ORDER)

def run_vector_collision_agent(client, message, ai_type, num_cycles):
    if ai_type == "claude" or ai_type == "chatgpt" or ai_type == "glm":
        full_message = [
            {"role": "user", "content": message}
        ]
    elif ai_type == "gemini":
        full_message = [
            {"role": "user", "parts": [{"text": message}]}
        ]
    else:
        raise ValueError("ai_type needs to be claude/chatgpt/gemini/glm")
    
    transcript = []
    remaining_cycles = num_cycles
    cycle_num = 1
    phase_idx = 0
    while True:
        if phase_idx % 3 == 0:
            print(f"\n ===== Cycle {cycle_num} / {num_cycles} ===== ")

        phase = ORDER[phase_idx]

        alert = (f"\nSystem Alert: cycle {cycle_num}/{num_cycles}, "
                 f"phase = {phase.value}. {remaining_cycles} cycles remain.")
        if ai_type == "gemini":
            full_message[-1]["parts"][0]["text"] += alert
        else:
            full_message[-1]["content"] += alert

        if phase == Phase.REVIEW_B:
            review_a_text = next(
                t["output"] for t in reversed(transcript) if t["phase"] == "review_a"
            )
            response, attempt_log = run_review_b_with_retries(
                client, full_message, ai_type, review_a_text, cycle_num
            )
            transcript.append({
                "phase": phase.value, "cycle": cycle_num,
                "output": response, "retry_log": attempt_log,
            })
        else:
            response = ai_outputs(client, full_message, ai_type, system_prompt=PROMPTS[phase])
            transcript.append({"phase": phase.value, "cycle": cycle_num, "output": response})
            print(f"\n--- cycle {cycle_num} / {phase.value} ---\n{response}")

            if ai_type == "gemini":
                full_message.append({"role": "model", "parts": [{"text": response}]})
            else:
                full_message.append({"role": "assistant", "content": response})

        if phase == Phase.GENERATE:
            if "Final Answer:" in response:
                _, _, final = response.partition("Final Answer:")
                return {"final_answer": final.strip(), "transcript": transcript}
            remaining_cycles -= 1
            cycle_num += 1
            if remaining_cycles <= 0:
                return {"final_answer": None, "transcript": transcript,
                        "note": "ran out of cycles without a Final Answer"}

        cue = NEXT_CUE[phase]
        if ai_type == "gemini":
            full_message.append({"role": "user", "parts": [{"text": cue}]})
        else:
            full_message.append({"role": "user", "content": cue})

        phase_idx = (phase_idx + 1) % len(ORDER)

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

user_message = """
Hi my name is Rish, What should I do today?
"""

result = run_normal_agent(client, user_message, "claude", 3)
print(result["final_answer"])

