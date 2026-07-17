from datasets import load_dataset
import base64, zlib, pickle, json, subprocess, sys

ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_latest", trust_remote_code=True)
split = ds["test"]

# ----- pick difficulty here: "easy", "medium", or "hard" -----
DIFFICULTY = "easy"

filtered = split.filter(lambda x: x["difficulty"] == DIFFICULTY)

# print available hard problems so you can pick one (Codeforces-sourced ones
# tend to be the toughest within the "hard" label — heavier on algorithms
# like segment trees, DP on trees, number theory, vs. LeetCode's pattern-based hards)
for i in range(len(filtered)):
    print(i, filtered[i]["question_title"], "-", filtered[i].get("platform"))

problem = filtered[0]  # change this index to try a different one

print("===== PROBLEM (feed this to the AI) =====")
print(problem["question_content"])

public_tests = json.loads(problem["public_test_cases"])
private_raw = pickle.loads(zlib.decompress(base64.b64decode(problem["private_test_cases"])))
private_tests = json.loads(private_raw) if isinstance(private_raw, str) else private_raw

print("\n===== PUBLIC TEST CASES =====")
print(public_tests)


import subprocess, sys, textwrap


def run_program(code: str, stdin_text: str):
    code = textwrap.dedent(code)  # strips leading indentation if you pasted code inside a function
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=stdin_text, capture_output=True, text=True, timeout=10,
    )
    return result.stdout, result.stderr


def check(code: str, tests: list, label: str):
    passed = 0
    for i, tc in enumerate(tests):
        stdout, stderr = run_program(code, tc["input"])
        actual = stdout.strip()
        expected = tc["output"].strip()
        ok = actual == expected
        passed += ok
        print(f"{label} test {i}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  expected: {expected!r}")
            print(f"  actual:   {actual!r}")
            if stderr:
                print(f"  stderr:   {stderr.strip()}")
    print(f"{label}: {passed}/{len(tests)} passed\n")


# ----- paste the model's returned code here as a string -----
model_code = """
t = int(input())
for _ in range(t):
    s = input()
    print("YES" if s == "abc" or sum(1 for i in range(3) if s[i] != "abc"[i]) == 2 else "NO")
"""

print("\n===== TESTING AGAINST PUBLIC CASES =====")
#check(model_code, public_tests, "public")

print("===== TESTING AGAINST PRIVATE CASES =====")
#check(model_code, private_tests, "private")