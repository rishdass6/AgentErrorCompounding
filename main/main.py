import os
from google.genai import types
from functions import text_vectorize_score, ai_outputs
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import anthropic
from dotenv import load_dotenv
from enum import Enum

load_dotenv(override=True)

model_text = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

print(f"All Models and Nessarcy imports loaded... \n")


REVIEW_A_SYS_PROMPT = """
You are Reviewer A in a Review A -> Review B -> Generate loop.
This turn ONLY: read the task (and, if present, the most recent Generate
output) and produce exactly 3 distinct, actionable claims about [AXIS ONE
-- e.g. correctness / logic]. Do not draft a revision. Do not touch [AXIS
TWO]. Output only:

Review A:
1. ...
2. ...
3. ...
"""

REVIEW_B_SYS_PROMPT = """
You are Reviewer B in the same loop. Review A's claims are already above
you in this conversation -- read them, don't repeat them.
This turn ONLY: produce exactly 3 distinct, actionable claims about
[AXIS TWO -- e.g. formatting / completeness]. Do not draft a revision.
Output only:

Review B:
1. ...
2. ...
3. ...
"""

GENERATE_SYS_PROMPT = """
You are the Generator. Review A's and Review B's claims for this cycle
are already above you.
This turn ONLY: produce a revised draft that visibly addresses all 6
claims. Never repeat the previous draft verbatim.
Unless the system alert says 0 cycles remain, output:

Generate:
[revised draft]

On the final cycle, output instead:

Final Answer:
[revised draft]
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

def run_agent(client, message, ai_type, num_cycles):
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

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

user_message = """
Hi my name is Rish, What should I do today?
"""

result = run_agent(client, user_message, "claude", 3)
print(result["final_answer"])

