"""
候选补召回实验（eval_recall_rerank.py）的纯逻辑测试。

覆盖：并集去重 / BM25 独有候选补算真实距离（不伪造）/ 0.85 过滤 /
      候选统计 / 可靠性规则不被绕过。

设计约束：不加载 embedding 与 Cross-Encoder 模型、不触碰正式数据库。
运行：pytest test_recall_rerank.py -v
"""

import pytest

from eval_recall_rerank import build_union, cosine_distance

CORPUS = [
    {"id": "id0", "text": "The intern reports to Director Mr Khor Kai Dat.",
     "source": "a.pdf", "page": 1},
    {"id": "id1", "text": "You are accepted as AI Programmer to develop AI Server.",
     "source": "a.pdf", "page": 1},
    {"id": "id2", "text": "Your allowance is RM1,000.00 per month.",
     "source": "b.pdf", "page": 2},
    {"id": "id3", "text": "Confidential information stays strictly confidential.",
     "source": "b.pdf", "page": 2},
]

# 手工构造的 2 维向量，便于验证 cosine 距离
# 单位方向：x=(1,0) y=(0,1) z=(1,1)/√2 ≈ (0.707,0.707) 对角
EMB = [
    [1.0, 0.0],   # id0
    [0.0, 1.0],   # id1
    [0.707, 0.707],  # id2（与对角查询距离 ≈ 0.293）
    [0.707, 0.707],  # id3
]


class TestCosineDistance:
    def test_identical_vectors_zero(self):
        assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_one(self):
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_diagonal_half(self):
        d = cosine_distance([1.0, 0.0], [0.707, 0.707])
        assert d == pytest.approx(1 - 0.7071, abs=1e-3)

    def test_symmetric(self):
        a, b = [0.3, 0.9], [0.8, -0.4]
        assert cosine_distance(a, b) == pytest.approx(cosine_distance(b, a))


class TestBuildUnion:
    def test_union_dedup_keeps_vector_distance(self):
        """BM25 与向量重合的候选去重，保留向量侧原始 distance（不覆盖）。"""
        vector_pairs = [(0, 0.40), (1, 0.55)]  # id0, id1 在向量 Top-10
        bm25_ids = [0, 2]                      # id0 重复、id2 是 BM25 独有
        query_emb = [1.0, 0.0]                 # 查询向量 = x 方向
        filtered, extra = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.85
        )
        ids = {c["id"] for c in filtered}
        assert ids == {"id0", "id1", "id2"}
        # 去重后的 id0 保留向量原始 distance 0.40
        d0 = next(c for c in filtered if c["id"] == "id0")
        assert d0["distance"] == pytest.approx(0.40)

    def test_bm25_only_candidate_gets_real_distance(self):
        """BM25 独有候选的 distance 来自真实 cosine 计算，不是伪造值。"""
        vector_pairs = [(0, 0.40)]
        bm25_ids = [1]  # id1 与查询正交 → 真实距离 ≈ 1.0
        query_emb = [1.0, 0.0]
        filtered, extra = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.85
        )
        assert len(extra) == 1
        d1 = extra[0]
        assert d1["id"] == "id1"
        assert d1["distance"] == pytest.approx(1.0)  # 真实计算值
        assert d1["source"] == "a.pdf" and d1["page"] == 1  # 元数据保留
        assert d1["id"] not in {i for i, _ in vector_pairs}  # 确认是独有候选

    def test_reliability_filter_applies_to_all(self):
        """0.85 过滤对向量侧与 BM25 补入候选一视同仁（不绕过）。"""
        vector_pairs = [(0, 0.40), (1, 0.90)]  # id1 向量距离已超阈值
        bm25_ids = [3]                          # id3 与对角查询距离 ≈ 0.293
        query_emb = [0.707, 0.707]
        filtered, extra = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.85
        )
        ids = {c["id"] for c in filtered}
        assert ids == {"id0", "id3"}   # id1(0.90) 被剔除、id3(≈0.29) 保留
        assert all(c["distance"] <= 0.85 for c in filtered)

    def test_all_bm25_only_above_threshold_removed(self):
        """BM25 独有候选距离全超阈值 → 被剔除，方案 C 不强行放行。"""
        vector_pairs = [(0, 0.40)]
        bm25_ids = [1, 2, 3]
        query_emb = [1.0, 0.0]  # id1 正交(1.0)、id2/id3 对角(≈0.293)
        filtered, extra = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.1
        )
        # 阈值 0.1：向量侧 id0(0.40) 与全部 BM25 独有候选都被剔除
        assert filtered == []
        assert {c["id"] for c in extra} == {"id1", "id2", "id3"}

    def test_no_duplicate_candidates_in_output(self):
        vector_pairs = [(0, 0.4), (1, 0.5), (2, 0.6)]
        bm25_ids = [0, 1, 2, 3]
        query_emb = [1.0, 0.0]
        filtered, extra = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.85
        )
        ids = [c["id"] for c in filtered]
        assert len(ids) == len(set(ids))  # 无重复候选

    def test_empty_inputs(self):
        assert build_union([], [], CORPUS, [1.0, 0.0], EMB, 0.85) == ([], [])

    def test_candidate_count_upper_bound(self):
        """并集候选数 <= 向量数 + BM25 数（去重后不会膨胀）。"""
        vector_pairs = [(i, 0.5) for i in range(4)]
        bm25_ids = [0, 1, 2, 3]
        query_emb = [1.0, 0.0]
        filtered, _ = build_union(
            vector_pairs, bm25_ids, CORPUS, query_emb, EMB, threshold=0.85
        )
        assert len(filtered) <= 4 + 4  # 理论上限
