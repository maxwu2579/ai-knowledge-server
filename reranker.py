"""
Cross-Encoder 重排（进程级单例 + lazy loading + 失败回退）。

模型：cross-encoder/ms-marco-MiniLM-L-6-v2（只推理，不训练/微调）。
- 第一次需要重排时加载一次，后续请求复用（线程锁保证并发不重复加载）；
- 模型缓存位于用户目录 ~/.cache/huggingface（项目外，不进入 Git）；
- 加载或推理失败：记录不含敏感信息的 warning，调用方回退纯向量排序
  （本模块直接返回原顺序，由 store.search 交给上层）。

本模块不依赖 store.py；candidates 是带 text / source / page / distance 的
dict 列表，重排只改变顺序，不改任何字段。
"""

import logging
import threading

from langfuse import get_client, observe

logger = logging.getLogger("reranker")

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_load_lock = threading.Lock()
_scorer = None
_load_failed = False  # 加载失败后不再反复尝试


def get_scorer():
    """进程级单例：返回 CrossEncoder 实例；加载失败返回 None。"""
    global _scorer, _load_failed
    if _scorer is not None:
        return _scorer
    if _load_failed:
        return None
    with _load_lock:
        if _scorer is not None:
            return _scorer  # 并发等待的线程直接复用
        try:
            from sentence_transformers import CrossEncoder

            _scorer = CrossEncoder(CE_MODEL)
            logger.info("Cross-Encoder 已加载：%s", CE_MODEL)
        except Exception as e:
            _load_failed = True
            logger.warning(
                "Cross-Encoder 加载失败（%s），本次及后续请求回退纯向量排序",
                type(e).__name__,
            )
            return None
    return _scorer


def reset() -> None:
    """清空单例（仅供测试）。"""
    global _scorer, _load_failed
    _scorer = None
    _load_failed = False


def warm_up_reranker() -> bool:
    """预热现有 Cross-Encoder singleton；失败时保留向量排序回退行为。"""
    scorer = get_scorer()
    if scorer is None:
        logger.warning(
            "Cross-Encoder warm-up failed; starting in vector-only fallback mode"
        )
        return False

    try:
        scorer.predict(
            [("startup warmup query", "startup warmup passage")],
            show_progress_bar=False,
        )
    except Exception as e:
        logger.warning(
            "Cross-Encoder warm-up prediction failed (%s); "
            "request-time fallback behavior remains active",
            type(e).__name__,
        )
        return False
    return True


@observe(name="rerank")
def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """用 Cross-Encoder 对候选重排，返回前 top_k（缺省返回全部）。

    - 空候选返回 []；
    - 模型缺失/加载失败/推理失败：返回原顺序（回退纯向量排序），
      不抛异常（上层 /search、/query 不会因此 500）；
    - 每项保留 text / source / page / distance 原字段，仅调整顺序。
    """
    if not candidates:
        get_client().update_current_span(
            metadata={"candidate_count": 0, "result_count": 0, "fallback": False}
        )
        return []

    scorer = get_scorer()
    if scorer is None:
        result = candidates[:top_k] if top_k is not None else candidates
        get_client().update_current_span(
            metadata={
                "candidate_count": len(candidates),
                "result_count": len(result),
                "fallback": True,
                "model": CE_MODEL,
            }
        )
        return result

    pairs = [(query, c["text"]) for c in candidates]
    try:
        scores = list(scorer.predict(pairs, show_progress_bar=False))
    except Exception as e:
        logger.warning(
            "Cross-Encoder 推理失败（%s），回退纯向量排序",
            type(e).__name__,
        )
        result = candidates[:top_k] if top_k is not None else candidates
        get_client().update_current_span(
            metadata={
                "candidate_count": len(candidates),
                "result_count": len(result),
                "fallback": True,
                "model": CE_MODEL,
            }
        )
        return result

    ranked = sorted(
        zip(candidates, scores), key=lambda x: x[1], reverse=True
    )
    result = [c for c, _ in ranked]
    result = result[:top_k] if top_k is not None else result
    get_client().update_current_span(
        metadata={
            "candidate_count": len(candidates),
            "result_count": len(result),
            "fallback": False,
            "model": CE_MODEL,
        }
    )
    return result
