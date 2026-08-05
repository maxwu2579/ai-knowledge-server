"""
重排接入（reranker.py + store.search）的测试。

设计约束：不加载真实 Cross-Encoder 模型（patch 注入假 scorer）、
不触碰正式 chroma_data_v2（store.get_collection 全部 mock）。

覆盖：Top-10 召回后重排 / top_k 截断 / 元数据与 distance 保持 /
      同分稳定排序 / 空候选 / 低于可靠性要求返回空 / 模型只加载一次 /
      并发不重复加载 / 加载失败回退 / 推理失败回退 /
      /search 不调用 DeepSeek / /query 来源使用重排顺序。
"""

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest


def make_candidate(text: str, source: str = "a.pdf", page: int = 1, distance: float = 0.5):
    return {"text": text, "source": source, "page": page, "distance": distance}


# ---------------------------------------------------------------------------
# reranker.py：单例 / lazy loading / 并发 / 回退
# ---------------------------------------------------------------------------


class TestScorerSingleton:
    def setup_method(self):
        import reranker

        reranker.reset()

    def teardown_method(self):
        import reranker

        reranker.reset()

    def test_get_scorer_loads_once(self):
        import reranker
        import sentence_transformers

        with patch.object(sentence_transformers, "CrossEncoder") as mce:
            mce.return_value = object()
            s1 = reranker.get_scorer()
            s2 = reranker.get_scorer()
            s3 = reranker.get_scorer()
            assert s1 is s2 is s3
            assert mce.call_count == 1  # 进程级单例：只加载一次

    def test_concurrent_calls_do_not_reload(self):
        import reranker
        import sentence_transformers

        with patch.object(sentence_transformers, "CrossEncoder") as mce:
            mce.return_value = object()
            with ThreadPoolExecutor(max_workers=8) as ex:
                scorers = list(ex.map(lambda _: reranker.get_scorer(), range(8)))
            assert all(s is scorers[0] for s in scorers)
            assert mce.call_count == 1  # 并发不重复加载

    def test_load_failure_returns_none_and_logs_warning(self, caplog):
        import reranker
        import sentence_transformers

        with patch.object(sentence_transformers, "CrossEncoder",
                          side_effect=RuntimeError("no torch")):
            with caplog.at_level(logging.WARNING, logger="reranker"):
                assert reranker.get_scorer() is None
        assert "加载失败" in caplog.text
        assert "no torch" not in caplog.text  # warning 不含敏感/原始异常细节

    def test_load_failure_not_retried(self):
        """加载失败后标记 _load_failed，后续调用不再反复尝试加载。"""
        import reranker
        import sentence_transformers

        with patch.object(sentence_transformers, "CrossEncoder",
                          side_effect=RuntimeError("boom")) as mce:
            assert reranker.get_scorer() is None
            assert reranker.get_scorer() is None
            assert mce.call_count == 1


# ---------------------------------------------------------------------------
# reranker.py：重排行为（假 scorer，不加载真实模型）
# ---------------------------------------------------------------------------


def fake_scorer(scores):
    class _Fake:
        def predict(self, pairs, show_progress_bar=False):
            return scores

    return _Fake()


class TestRerankBehavior:
    def test_orders_by_score_descending(self):
        import reranker

        cands = [
            make_candidate("alpha", distance=0.6),
            make_candidate("beta", distance=0.4),
            make_candidate("gamma", distance=0.5),
        ]
        with patch("reranker.get_scorer",
                   return_value=fake_scorer([1.0, 5.0, 3.0])):
            ranked = reranker.rerank("q", cands)
        assert [c["text"] for c in ranked] == ["beta", "gamma", "alpha"]

    def test_tie_scores_keep_original_order(self):
        import reranker

        cands = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        with patch("reranker.get_scorer", return_value=fake_scorer([2.0, 2.0, 2.0])):
            ranked = reranker.rerank("q", cands)
        assert [c["text"] for c in ranked] == ["a", "b", "c"]  # 稳定排序

    def test_metadata_and_distance_preserved(self):
        import reranker

        cands = [
            make_candidate("x", source="s1.pdf", page=2, distance=0.71),
            make_candidate("y", source="s2.pdf", page=3, distance=0.62),
        ]
        with patch("reranker.get_scorer", return_value=fake_scorer([0.1, 9.9])):
            ranked = reranker.rerank("q", cands)
        assert ranked[0] == cands[1]  # 重排不生成新对象、不改字段
        assert ranked[1] == cands[0]
        assert ranked[0]["source"] == "s2.pdf" and ranked[0]["page"] == 3
        assert ranked[0]["distance"] == 0.62  # distance 仍是原始向量距离

    def test_top_k_truncation(self):
        import reranker

        cands = [make_candidate(f"d{i}") for i in range(5)]
        with patch("reranker.get_scorer",
                   return_value=fake_scorer([float(i) for i in range(5)])):
            ranked = reranker.rerank("q", cands, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["text"] == "d4"

    def test_empty_candidates_returns_empty(self):
        import reranker

        assert reranker.rerank("q", []) == []

    def test_inference_failure_falls_back_to_original_order(self, caplog):
        import reranker

        cands = [make_candidate("a", distance=0.4), make_candidate("b", distance=0.6)]
        class _Boom:
            def predict(self, pairs, show_progress_bar=False):
                raise RuntimeError("infer boom")

        with patch("reranker.get_scorer", return_value=_Boom()):
            with caplog.at_level(logging.WARNING, logger="reranker"):
                ranked = reranker.rerank("q", cands)
        assert [c["text"] for c in ranked] == ["a", "b"]  # 原向量顺序回退
        assert "推理失败" in caplog.text

    def test_scorer_missing_falls_back_to_original_order(self):
        import reranker

        cands = [make_candidate("a", distance=0.4), make_candidate("b", distance=0.6)]
        with patch("reranker.get_scorer", return_value=None):
            ranked = reranker.rerank("q", cands)
        assert [c["text"] for c in ranked] == ["a", "b"]


# ---------------------------------------------------------------------------
# store.search：Top-10 召回 → 0.85 阈值 → 重排（store 层全部 mock）
# ---------------------------------------------------------------------------


class FakeCollection:
    """模拟 ChromaDB collection：query 遵守 n_results 截断（与真实行为一致）。"""

    def __init__(self, docs, metas, dists, ids, count):
        self._docs = docs
        self._metas = metas
        self._dists = dists
        self._ids = ids
        self._count = count
        self.query_call_count = 0

    def count(self):
        return self._count

    def query(self, query_texts=None, n_results=None):
        self.query_call_count += 1
        n = min(n_results or len(self._docs), len(self._docs))
        return {
            "documents": [self._docs[:n]],
            "metadatas": [self._metas[:n]],
            "distances": [self._dists[:n]],
            "ids": [self._ids[:n]],
        }


def fake_collection(docs, metas, dists, ids, count):
    return FakeCollection(docs, metas, dists, ids, count)


class TestStoreSearchRerank:
    def test_recalls_at_most_10_then_reranks(self):
        import store

        docs = [f"doc{i}" for i in range(15)]
        metas = [{"source": "a.pdf", "page": 1}] * 15
        dists = [0.5 + i * 0.03 for i in range(15)]  # 0.50 ~ 0.92
        ids = [f"id{i}" for i in range(15)]
        expected = [make_candidate(docs[1], distance=dists[1])]

        with (
            patch("store.get_collection", return_value=fake_collection(
                docs, metas, dists, ids, count=15)),
            patch("store.rerank", return_value=expected) as mock_rerank,
        ):
            hits = store.search("the question", top_k=3)

        mock_rerank.assert_called_once()
        query_arg, cands_arg = mock_rerank.call_args.args
        assert query_arg == "the question"
        assert len(cands_arg) == 10  # 向量召回上限 Top-10（0.50~0.77 全过阈值）
        assert all(c["distance"] <= 0.85 for c in cands_arg)  # 阈值先于重排
        assert mock_rerank.call_args.kwargs["top_k"] == 3
        assert hits == expected  # 返回值即重排结果

    def test_threshold_filters_before_rerank(self):
        """距离 > 0.85 的候选被剔除，不进入重排。"""
        import store

        docs = ["far", "near"]
        metas = [{"source": "a.pdf", "page": 1}] * 2
        dists = [0.9, 0.5]  # far 超过阈值
        ids = ["id0", "id1"]

        with (
            patch("store.get_collection", return_value=fake_collection(
                docs, metas, dists, ids, count=2)),
            patch("store.rerank", return_value=[make_candidate("near", distance=0.5)])
            as mock_rerank,
        ):
            store.search("q", top_k=5)
        cands = mock_rerank.call_args.args[1]
        assert [c["text"] for c in cands] == ["near"]

    def test_all_above_threshold_returns_empty(self):
        """全部候选 distance > 0.85 → 返回 []（无可靠结果），且不调用重排。"""
        import store

        docs = ["x", "y"]
        metas = [{"source": "a.pdf", "page": 1}] * 2
        dists = [0.9, 0.95]
        ids = ["id0", "id1"]

        with (
            patch("store.get_collection", return_value=fake_collection(
                docs, metas, dists, ids, count=2)),
            patch("store.rerank") as mock_rerank,
        ):
            assert store.search("q", top_k=5) == []
        mock_rerank.assert_not_called()

    def test_empty_collection_returns_empty(self):
        import store

        col = MagicMock()
        col.count.return_value = 0
        with (
            patch("store.get_collection", return_value=col),
            patch("store.rerank") as mock_rerank,
        ):
            assert store.search("q") == []
        mock_rerank.assert_not_called()

    def test_repeated_search_uses_live_database(self):
        """每次 search 都重新查库（不维护过期候选缓存）。"""
        import store

        col = MagicMock()
        col.count.return_value = 3
        docs = ["a", "b", "c"]
        metas = [{"source": "a.pdf", "page": 1}] * 3
        dists = [0.5, 0.6, 0.7]
        ids = ["id0", "id1", "id2"]
        col.query.return_value = {
            "documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids],
        }
        with (
            patch("store.get_collection", return_value=col),
            patch("store.rerank", side_effect=lambda q, c, top_k=None: c[:top_k]),
        ):
            store.search("q1")
            store.search("q2")
        assert col.query.call_count == 2  # 两次查询都打到数据库，无缓存


# ---------------------------------------------------------------------------
# API 层：/search 零 DeepSeek；/query 来源使用重排顺序
# ---------------------------------------------------------------------------


class TestApiWithRerank:
    def test_search_still_does_not_call_deepseek(self):
        """重排接入后 /search 仍不调用 ask / rewrite_query。"""
        from fastapi.testclient import TestClient
        from api import app
        from ask import rewrite_query

        with (
            patch("api.vector_search", return_value=[
                {"text": "16 weeks", "source": "a.pdf", "page": 1, "distance": 0.5},
            ]),
            patch("api.ask") as mock_ask,
            patch("ask.rewrite_query") as mock_rewrite,
        ):
            resp = TestClient(app).post("/search", json={"query": "实习期多长？"})
            assert resp.status_code == 200
            mock_ask.assert_not_called()
            mock_rewrite.assert_not_called()

    def test_query_sources_follow_rerank_order(self):
        """/query 返回的 sources 顺序 = 检索函数（重排后）的顺序。"""
        from fastapi.testclient import TestClient
        from api import app

        reranked = [
            {"text": "second best", "source": "b.pdf", "page": 1, "distance": 0.6},
            {"text": "best match", "source": "a.pdf", "page": 2, "distance": 0.7},
            {"text": "third", "source": "c.pdf", "page": 1, "distance": 0.8},
        ]
        with (
            patch("api.ask") as mock_ask,
            patch("api.stats", return_value={"chunks": 5, "sources": ["a.pdf"]}),
        ):
            mock_ask.return_value = {
                "answer": "答案 [b.pdf p.1]",
                "sources": reranked,
            }
            resp = TestClient(app).post("/query", json={"question": "实习期是多久？"})
            assert resp.status_code == 200
            sources = resp.json()["sources"]
            assert [s["source"] for s in sources] == ["b.pdf", "a.pdf", "c.pdf"]
            assert [s["page"] for s in sources] == [1, 2, 1]
            assert [s["distance"] for s in sources] == [0.6, 0.7, 0.8]

    def test_upload_still_works_with_rerank_store(self):
        """重排接入不破坏上传流程（夹具 mock 的切块/入库路径不变）。"""
        from fastapi.testclient import TestClient
        from api import app

        with (
            patch("api.add_chunks", return_value=2),
            patch("api.delete_source"),
            patch("api.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("api.load_document_paragraphs") as mock_paras,
        ):
            from chunker import Chunk

            mock_paras.return_value = [
                Chunk(text="p1", source="note.txt", page=1),
                Chunk(text="p2", source="note.txt", page=1),
            ]
            resp = TestClient(app).post(
                "/documents/upload",
                files={"file": ("note.txt", io.BytesIO(b"data " * 20), "text/plain")},
            )
            assert resp.status_code == 200
            assert resp.json()["chunks"] == 2
