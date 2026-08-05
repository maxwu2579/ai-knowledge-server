"""
DeepSeek 自动查询改写（仅评估用，不进入正式服务）。

把中文/中英混合问题改写成简洁、自然的英文检索语句；英文问题不调用。
改写结果缓存到 eval_rewrite_cache.json，重复运行不会重复产生 API 费用。

用法：
    py eval_rewrite.py          # 只改写缓存中缺失的问题，并打印全部改写供检查
    py eval_rewrite.py --force  # 忽略缓存全部重写
    py eval_rewrite.py --show   # 只打印缓存，不调用 API

约束：
- 改写只保留原问题含义：禁止回答问题、补充事实或加入标准答案；
- 自动校验：改写不得含中文、不得包含标准答案片段（疑似泄漏），违规跳过；
- 不修改 store.py / api.py / .env / chroma_data；
- /search 不接入本模块（保持零 DeepSeek API 费用）。
"""

import argparse
import json
import os
import re
from pathlib import Path

import httpx

import ask  # noqa: F401  # 触发 ask 模块加载 .env（只读，不修改）
from eval_questions import QUESTIONS

CACHE_FILE = Path(__file__).parent / "eval_rewrite_cache.json"
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-pro")

SYSTEM_PROMPT = """You translate search queries for a document retrieval system.

Rules:
- Translate the user's search query into a concise, natural English retrieval query.
- Preserve the original meaning only. Do NOT answer the question.
- Do NOT add facts, numbers, names or details that are not in the query itself.
- Keep proper nouns that appear in the query (names, English terms) unchanged.
- Output ONLY the English translation — no quotes, no explanation, no extra punctuation."""

CJK_RE = re.compile(r"[一-鿿]")


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------


def load_cache(path: Path = CACHE_FILE) -> dict:
    """读取改写缓存 {问题: 英文改写}。"""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path = CACHE_FILE) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 改写校验（防泄漏：不改写=不回答问题）
# ---------------------------------------------------------------------------


def validate_rewrite(question: str, rewritten: str) -> str:
    """
    校验自动改写：非空、纯英文、不包含标准答案片段。违规抛 ValueError。

    返回去除首尾引号后的改写文本。
    """
    text = rewritten.strip().strip('"').strip("'").strip()
    if not text:
        raise ValueError(f"改写为空: {question!r}")
    if CJK_RE.search(text):
        raise ValueError(f"改写仍含中文: {question!r} -> {text!r}")
    q = next((x for x in QUESTIONS if x["question"] == question), None)
    if q is not None and q["expected_fragment"] in text:
        raise ValueError(f"改写包含标准答案片段（疑似泄漏）: {question!r} -> {text!r}")
    return text


# ---------------------------------------------------------------------------
# DeepSeek 调用
# ---------------------------------------------------------------------------


def call_deepseek(question: str) -> str:
    """调用 DeepSeek 把一个问题改写为英文，返回原始回复文本。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法自动改写。")

    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "temperature": 0,
            },
            timeout=60,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"调用 DeepSeek 失败: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek 返回 HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"解析 DeepSeek 响应失败: {e}")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def print_all(cache: dict) -> None:
    """打印全部中文/混合问题的改写，供人工检查泄漏与语义变化。"""
    lang_label = {"zh": "中文", "mixed": "混合"}
    print("== 自动改写结果（中文/混合，供检查泄漏与语义）==")
    for q in QUESTIONS:
        if q["lang"] not in ("zh", "mixed"):
            continue
        rw = cache.get(q["question"])
        status = f'"{rw}"' if rw else "(未生成)"
        print(f"  [{lang_label[q['lang']]}] {q['question']} -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略缓存全部重写")
    parser.add_argument("--show", action="store_true", help="只打印缓存，不调用 API")
    args = parser.parse_args()

    if args.show:
        print_all(load_cache())
        return

    cache = {} if args.force else load_cache()
    targets = [q for q in QUESTIONS if q["lang"] in ("zh", "mixed")]
    todo = [q for q in targets if q["question"] not in cache]
    print(f"需要改写 {len(todo)}/{len(targets)} 题，缓存命中 {len(targets) - len(todo)} 题")

    for q in todo:
        try:
            text = validate_rewrite(q["question"], call_deepseek(q["question"]))
        except Exception as e:
            print(f"  跳过 {q['question']!r}: {e}")
            continue
        cache[q["question"]] = text
        save_cache(cache)  # 每题一存，中途中断也不丢
        print(f"  ✓ {q['question']} -> {text}")

    print_all(cache)


if __name__ == "__main__":
    main()
