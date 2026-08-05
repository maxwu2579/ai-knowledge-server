"""
API 自动化测试。

运行：
    pytest test_api.py -v
"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_store_and_chunker():
    """
    全局 mock：屏蔽 ChromaDB / embedding / LLM 等外部依赖。
    所有函数 patch 在 api 模块的命名空间中（即 api 通过 import 持有的引用）。
    注意：api 的正式上传切块入口是方案C的 load_document_paragraphs。
    """
    with (
        patch("api.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
        patch("api.add_chunks", return_value=3),
        patch("api.delete_source"),
        patch("api.load_document_paragraphs") as mock_paragraphs,
    ):
        # load_document_paragraphs 返回 3 个假 chunk
        mock_paragraphs.return_value = [
            __import__("chunker").Chunk(text="a", source="test.pdf", page=1),
            __import__("chunker").Chunk(text="b", source="test.pdf", page=1),
            __import__("chunker").Chunk(text="c", source="test.pdf", page=1),
        ]
        yield


@pytest.fixture
def client():
    from api import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert isinstance(body["chunks"], int)
        assert isinstance(body["sources"], list)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_empty_question_returns_422(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_whitespace_question_returns_422(self, client):
        resp = client.post("/query", json={"question": "   \t\n  "})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_no_relevant_results_skips_llm(self, client):
        """DB 有数据但无 chunk 通过阈值时，不调 LLM，answer 提示找不到。"""
        with patch("api.ask") as mock_ask:
            mock_ask.return_value = {
                "answer": "资料中找不到这个问题的答案。",
                "sources": [],
            }
            resp = client.post("/query", json={"question": "xyzxyz完全不相关"})
            assert resp.status_code == 200
            assert "找不到" in resp.json()["answer"]

    def test_valid_question_returns_answer_and_sources(self, client):
        with patch("api.ask") as mock_ask:
            mock_ask.return_value = {
                "answer": "实习期16周 [test.pdf p.1]",
                "sources": [
                    {
                        "text": "16 weeks internship",
                        "source": "test.pdf",
                        "page": 1,
                        "distance": 0.45,
                    }
                ],
            }
            resp = client.post("/query", json={"question": "实习期多长"})
            assert resp.status_code == 200
            body = resp.json()
            assert "answer" in body
            assert len(body["sources"]) == 1
            assert body["sources"][0]["source"] == "test.pdf"

    # ---- LLM 异常 → HTTP 状态码 ------------------------------------------------

    def test_llm_auth_error_returns_502(self, client):
        from ask import LLMAuthError

        with patch("api.ask", side_effect=LLMAuthError("bad key")):
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 502

    def test_llm_rate_limit_returns_503(self, client):
        from ask import LLMRateLimitError

        with patch("api.ask", side_effect=LLMRateLimitError("too many")):
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 503

    def test_llm_server_error_returns_502(self, client):
        from ask import LLMServerError

        with patch("api.ask", side_effect=LLMServerError("boom")):
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 502

    def test_llm_connection_error_returns_502(self, client):
        from ask import LLMConnectionError

        with patch("api.ask", side_effect=LLMConnectionError("timeout")):
            resp = client.post("/query", json={"question": "test"})
            assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_valid_query_returns_results_without_llm(self, client):
        """正常检索：返回四个字段，且完全不调用 DeepSeek。"""
        with (
            patch("api.vector_search") as mock_search,
            patch("api.ask") as mock_ask,
        ):
            mock_search.return_value = [
                {
                    "text": "实习期为16周。",
                    "source": "university letter concerning d internship.pdf",
                    "page": 1,
                    "distance": 0.45,
                },
                {
                    "text": "实习期可以申请延长。",
                    "source": "university letter concerning d internship.pdf",
                    "page": 1,
                    "distance": 0.55,
                },
            ]
            resp = client.post("/search", json={"query": "  实习期是多久？  ", "top_k": 2})
            assert resp.status_code == 200
            body = resp.json()
            assert len(body) == 2
            assert body[0]["source"] == "university letter concerning d internship.pdf"
            assert body[0]["page"] == 1
            assert "实习期" in body[0]["text"]
            assert body[0]["distance"] == 0.45
            # query 已去除空白，原样传给向量检索；DeepSeek 未被调用
            mock_search.assert_called_once_with("实习期是多久？", top_k=2)
            mock_ask.assert_not_called()

    def test_default_top_k_is_5(self, client):
        """top_k 缺省时按 5 处理。"""
        with patch("api.vector_search") as mock_search:
            mock_search.return_value = []
            client.post("/search", json={"query": "实习"})
            mock_search.assert_called_once_with("实习", top_k=5)

    def test_blank_query_returns_422(self, client):
        resp = client.post("/search", json={"query": "   \t\n  "})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422

    def test_top_k_below_range_returns_422(self, client):
        resp = client.post("/search", json={"query": "实习", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_above_range_returns_422(self, client):
        resp = client.post("/search", json={"query": "实习", "top_k": 21})
        assert resp.status_code == 422

    def test_top_k_not_an_integer_returns_422(self, client):
        resp = client.post("/search", json={"query": "实习", "top_k": "abc"})
        assert resp.status_code == 422

    def test_no_relevant_results_returns_empty_array(self, client):
        """没有任何段落通过相关性阈值时返回空数组。"""
        with patch("api.vector_search", return_value=[]):
            resp = client.post("/search", json={"query": "完全不相关xyz"})
            assert resp.status_code == 200
            assert resp.json() == []

    def test_search_does_not_call_any_deepseek_function(self, client):
        """/search 不调用 ask 的回答、也不调用改写——即使 query 是中文。"""
        from ask import rewrite_query

        with (
            patch("api.vector_search", return_value=[]),
            patch("api.ask") as mock_ask,
            patch("ask.rewrite_query") as mock_rewrite,
        ):
            resp = client.post("/search", json={"query": "实习什么时候开始？"})
            assert resp.status_code == 200
            mock_ask.assert_not_called()
            mock_rewrite.assert_not_called()


# ---------------------------------------------------------------------------
# ask 模块：中文问题自动英文改写
# ---------------------------------------------------------------------------

FAKE_HIT = {
    "text": "16 WEEKS OF COMPULSORY INTERNSHIP",
    "source": "test.pdf",
    "page": 1,
    "distance": 0.4,
}


class TestAskRewrite:
    """ask() 的改写流程：中文改写一次、英文跳过、失败/空回退、答案用原文语言。"""

    def test_chinese_question_rewrites_once_then_searches_and_answers(self):
        from ask import REWRITE_SYSTEM_PROMPT, SYSTEM_PROMPT

        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [
                "When does the internship start?",  # 第 1 次：改写
                "实习从 2026 年 9 月 14 日开始。",   # 第 2 次：回答
            ]
            from ask import ask

            result = ask("实习什么时候开始？")

            assert mock_llm.call_count == 2
            # 第 1 次调用是改写：用改写提示词
            first = mock_llm.call_args_list[0]
            assert first.kwargs["system_prompt"] == REWRITE_SYSTEM_PROMPT
            # 第 2 次调用是回答：不传提示词（默认就是答案提示词），
            # 且问题保持用户原文（中文）
            second = mock_llm.call_args_list[1]
            assert second.kwargs.get("system_prompt", SYSTEM_PROMPT) == SYSTEM_PROMPT
            assert "实习什么时候开始" in second.args[0]
            # 检索用的是改写后的英文查询
            mock_search.assert_called_once_with("When does the internship start?", top_k=5)
            # 答案与来源原样返回
            assert result["answer"] == "实习从 2026 年 9 月 14 日开始。"
            assert result["sources"] == [FAKE_HIT]

    def test_english_question_skips_rewrite(self):
        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
        ):
            mock_llm.side_effect = ["The internship lasts 16 weeks."]
            from ask import ask

            result = ask("When does the internship start?")

            assert mock_llm.call_count == 1  # 只回答，不改写
            mock_search.assert_called_once_with("When does the internship start?", top_k=5)
            assert "When does the internship start?" in mock_llm.call_args_list[0].args[0]

    def test_rewrite_failure_falls_back_to_original_question(self):
        from ask import LLMAuthError

        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [LLMAuthError("no key"), "中文答案"]
            from ask import ask

            result = ask("实习什么时候开始？")

            # 改写失败：回退用原问题检索，回答照常
            mock_search.assert_called_once_with("实习什么时候开始？", top_k=5)
            assert result["answer"] == "中文答案"

    def test_rewrite_empty_string_falls_back_to_original_question(self):
        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
        ):
            mock_llm.side_effect = ["", "中文答案"]
            from ask import ask

            result = ask("实习什么时候开始？")

            mock_search.assert_called_once_with("实习什么时候开始？", top_k=5)
            assert result["answer"] == "中文答案"

    def test_rewrite_whitespace_falls_back_to_original_question(self):
        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
        ):
            mock_llm.side_effect = ["   \n  ", "中文答案"]
            from ask import ask

            ask("实习什么时候开始？")
            mock_search.assert_called_once_with("实习什么时候开始？", top_k=5)

    def test_chinese_answer_language_rule_in_system_prompt(self):
        """回答提示词要求语言跟随提问语言——中文提问应得到中文回答。"""
        from ask import SYSTEM_PROMPT

        assert "same language as the question" in SYSTEM_PROMPT

    def test_system_prompt_requires_english_answer_for_english_question(self):
        """回答阶段的生成提示必须明确要求：英文提问 → 英文回答。"""
        from ask import SYSTEM_PROMPT

        assert "If the question is in English, you MUST answer in English." in SYSTEM_PROMPT

    def test_system_prompt_requires_chinese_answer_for_chinese_question(self):
        """回答阶段的生成提示必须明确要求：中文提问 → 中文回答。"""
        from ask import SYSTEM_PROMPT

        assert "If the question is in Chinese, you MUST answer in Chinese." in SYSTEM_PROMPT

    def test_system_prompt_other_languages_follow_question(self):
        """回答阶段的生成提示对其他语言也要求尽量同语言回答。"""
        from ask import SYSTEM_PROMPT

        assert "other language" in SYSTEM_PROMPT
        assert "as much as possible" in SYSTEM_PROMPT

    def test_english_question_does_not_trigger_rewrite(self):
        """英文问题完全不触发 query rewrite：改写函数不被调用，只调一次回答。"""
        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
            patch("ask.rewrite_query") as mock_rewrite,
        ):
            mock_llm.side_effect = ["The internship lasts 16 weeks."]
            from ask import ask

            result = ask("How long is the internship?")

            mock_rewrite.assert_not_called()
            assert mock_llm.call_count == 1  # 只回答，不调用改写
            mock_search.assert_called_once_with("How long is the internship?", top_k=5)
            # 回答阶段用的是原始英文问题
            assert "How long is the internship?" in mock_llm.call_args_list[0].args[0]
            assert result["answer"] == "The internship lasts 16 weeks."

    def test_chinese_question_triggers_rewrite(self):
        """中文问题触发 query rewrite：改写后用英文查询检索，回答仍用原始中文问题。"""
        from ask import SYSTEM_PROMPT

        with (
            patch("ask.stats", return_value={"chunks": 5, "sources": ["test.pdf"]}),
            patch("ask.search", return_value=[FAKE_HIT]) as mock_search,
            patch("ask._call_llm") as mock_llm,
            patch("ask.rewrite_query", return_value="How long is the internship?") as mock_rewrite,
        ):
            mock_llm.side_effect = ["实习期为16周。"]  # 只有回答一次调用
            from ask import ask

            ask("实习期是多久？")

            mock_rewrite.assert_called_once_with("实习期是多久？")
            mock_search.assert_called_once_with("How long is the internship?", top_k=5)
            # 回答阶段不传提示词（默认即答案提示词），且用原始中文问题，而不是英文改写
            gen = mock_llm.call_args_list[0]
            assert gen.kwargs.get("system_prompt", SYSTEM_PROMPT) == SYSTEM_PROMPT
            assert "实习期是多久" in gen.args[0]
            assert "How long is the internship?" not in gen.args[0]

    def test_contains_chinese(self):
        from ask import contains_chinese

        assert contains_chinese("实习什么时候开始？")
        assert contains_chinese("WU ZHONGHENG 的 Student ID 是多少？")  # 混合也算中文问题
        assert not contains_chinese("When does the internship start?")
        assert not contains_chinese("internship period")
        assert not contains_chinese("")

    def test_rewrite_prompt_only_translates(self):
        """改写提示词必须明确：只翻译、不回答、不补充信息、单行输出。"""
        from ask import REWRITE_SYSTEM_PROMPT

        p = REWRITE_SYSTEM_PROMPT.lower()
        assert "translate" in p
        assert "do not answer" in p
        assert "do not add" in p
        assert "single line" in p


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_upload_binds_paragraph_chunker(self):
        """api.py 的正式上传切块入口是方案C函数（不再是旧的 load_document）。"""
        import api

        src = Path(api.__file__).read_text(encoding="utf-8")
        assert "from chunker import load_document_paragraphs" in src
        assert "chunks = load_document_paragraphs(tmp_path)" in src
        # 模块绑定：方案C入口存在，旧入口 load_document 不再被 api 使用
        assert hasattr(api, "load_document_paragraphs")
        assert not hasattr(api, "load_document")

    def test_pdf_and_txt_uploads_both_use_paragraph_chunking(self, client):
        """PDF 与 TXT 上传都走方案C切块入口（夹具 mock 的就是该入口）。"""
        for name, content, mime in (
            ("note.pdf", b"%PDF-1.4 fake pdf content here", "application/pdf"),
            ("note.txt", b"hello world " * 30, "text/plain"),
            ("note.md", b"# Hello\n\nSome content.", "text/markdown"),
        ):
            resp = client.post(
                "/documents/upload",
                files={"file": (name, io.BytesIO(content), mime)},
            )
            assert resp.status_code == 200, name
            assert resp.json()["chunks"] == 3, name

    def test_reupload_calls_delete_then_add(self, client):
        """同名文件重复上传：每次先删旧数据再入库（不残留、块数不叠加）。"""
        with patch("api.add_chunks") as mock_add, patch("api.delete_source") as mock_del:
            mock_add.return_value = 3
            for _ in range(2):
                resp = client.post(
                    "/documents/upload",
                    files={"file": ("again.txt", io.BytesIO(b"data " * 20), "text/plain")},
                )
                assert resp.status_code == 200
            # 两次上传各触发一次删除 + 一次入库
            assert mock_del.call_count == 2
            assert mock_add.call_count == 2
            mock_del.assert_called_with("again.txt")

    def test_unsupported_extension_returns_415(self, client):
        """不支持的文件类型返回 415（可靠性加固后）。"""
        resp = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.exe",
                    io.BytesIO(b"fake"),
                    "application/octet-stream",
                )
            },
        )
        assert resp.status_code == 415
        assert "不支持" in resp.json()["detail"]

    def test_supported_extension_succeeds(self, client):
        resp = client.post(
            "/documents/upload",
            files={
                "file": ("notes.txt", io.BytesIO(b"hello world " * 30), "text/plain")
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "notes.txt"
        assert body["chunks"] == 3
        assert "已导入" in body["message"]

    def test_path_traversal_in_filename_returns_400(self, client):
        resp = client.post(
            "/documents/upload",
            files={
                "file": (
                    "../../etc/passwd.pdf",
                    io.BytesIO(b"malicious"),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 400
        assert "非法" in resp.json()["detail"]

    def test_absolute_path_filename_returns_400(self, client):
        resp = client.post(
            "/documents/upload",
            files={
                "file": (
                    "/etc/shadow.txt",
                    io.BytesIO(b"malicious"),
                    "text/plain",
                )
            },
        )
        assert resp.status_code == 400
        assert "非法" in resp.json()["detail"]

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp.status_code == 400

    def test_pdf_supported(self, client):
        resp = client.post(
            "/documents/upload",
            files={
                "file": (
                    "report.pdf",
                    io.BytesIO(b"%PDF-1.4 fake pdf content here"),
                    "application/pdf",
                )
            },
        )
        assert resp.status_code == 200

    def test_markdown_supported(self, client):
        resp = client.post(
            "/documents/upload",
            files={
                "file": ("README.md", io.BytesIO(b"# Hello\n\nSome content."), "text/markdown")
            },
        )
        assert resp.status_code == 200

    def test_oversized_file_returns_413(self, client):
        """通过临时降低 MAX_UPLOAD_BYTES 来验证超大文件被拒绝。"""
        import api

        original = api.MAX_UPLOAD_BYTES
        try:
            api.MAX_UPLOAD_BYTES = 100  # 100 字节
            resp = client.post(
                "/documents/upload",
                files={
                    "file": ("big.pdf", io.BytesIO(b"x" * 5000), "application/pdf")
                },
            )
            assert resp.status_code == 413
            assert "过大" in resp.json()["detail"] or "最大" in resp.json()["detail"]
        finally:
            api.MAX_UPLOAD_BYTES = original


# ---------------------------------------------------------------------------
# secure_filename 独立测试
# ---------------------------------------------------------------------------


class TestSecureFilename:
    def test_path_traversal_raises_http_exception(self):
        """包含 ../ 的文件名直接拒绝，不截取。"""
        from api import HTTPException, secure_filename

        with pytest.raises(HTTPException) as exc:
            secure_filename("../../etc/passwd.pdf")
        assert exc.value.status_code == 400
        assert "非法" in exc.value.detail

    def test_absolute_path_raises_http_exception(self):
        """绝对路径被拒绝。"""
        from api import HTTPException, secure_filename

        with pytest.raises(HTTPException) as exc:
            secure_filename("/etc/shadow.txt")
        assert exc.value.status_code == 400

    def test_normal_filename_passes(self):
        from api import secure_filename

        assert secure_filename("my document (1).pdf") == "my document (1).pdf"

    def test_empty_name_raises(self):
        import pytest as pt
        from api import HTTPException, secure_filename

        with pt.raises(HTTPException) as exc:
            secure_filename("   ")
        assert exc.value.status_code == 400
