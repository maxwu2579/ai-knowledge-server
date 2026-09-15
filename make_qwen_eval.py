from pathlib import Path

scripts_dir = Path(__file__).parent / "evaluation" / "scripts"
src = scripts_dir / "eval_bge_small.py"
dst = scripts_dir / "eval_qwen_bge_small.py"

s = src.read_text(encoding="utf-8")

old = '''def pick_query(q: dict) -> str:
    return q.get("rewritten_en") or q["question"]'''

new = '''import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN_MODEL = "Qwen/Qwen3-0.6B"

print("Loading Qwen3-0.6B for local query rewrite...")
_qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
_qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    dtype=torch.float32
)
_qwen_model.eval()

_qwen_cache = {}
_qwen_rewrite_times = []

def pick_query(q: dict) -> str:
    question = q["question"]

    if question in _qwen_cache:
        return _qwen_cache[question]

    messages = [
        {
            "role": "system",
            "content": "Rewrite the user query into concise English for semantic retrieval. Output only the rewritten query. Do not explain."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    text = _qwen_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = _qwen_tok(text, return_tensors="pt")

    import time
    start = time.perf_counter()

    with torch.inference_mode():
        out = _qwen_model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=_qwen_tok.eos_token_id
        )

    elapsed = time.perf_counter() - start
    _qwen_rewrite_times.append(elapsed)

    rewrite = _qwen_tok.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    _qwen_cache[question] = rewrite
    return rewrite'''

if old not in s:
    raise SystemExit("pick_query block not found. No file was changed.")

s = s.replace(old, new, 1)

s += '''
if _qwen_rewrite_times:
    print()
    print("=== Qwen3-0.6B Rewrite Performance ===")
    print(f"Unique queries rewritten: {len(_qwen_rewrite_times)}")
    print(f"Average rewrite latency: {sum(_qwen_rewrite_times)/len(_qwen_rewrite_times):.4f} s/query")
'''

dst.write_text(s, encoding="utf-8")
print(f"{dst} created.")
