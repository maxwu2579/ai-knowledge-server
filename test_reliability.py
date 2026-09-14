"""
API 可靠性加固测试：上传限制 / 路径穿越 / 超时 / 重试条件 /
敏感信息脱敏 / 错误状态码 / request_id。

覆盖 ask.py（DeepSeek 调用）与 api.py（上传、中间件、异常处理）。
不调用真实 DeepSeek（httpx.post 全部 mock），不触碰正式数据库。
运行：pytest test_reliability.py -v
"""

import io
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ask import (
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMServerError,
)


# ---------------------------------------------------------------------------
# 工具：假 HTTP 响应
# ---------------------------------------------------------------------------


def fake_response(status: int, body: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.is_success = 200 <= status < 300
    r.text = json.dumps(body) if body is not None else ""
    if body is not None:
        r.json.return_value = body
    return r


OK_BODY = {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """所有 _call_llm 测试 patch 掉 httpx.post 与退避等待。"""
    with (
        patch("ask.httpx.post") as mock_post,
        patch("ask.time.sleep"),
    ):
        mock_post.side_effect = AssertionError(
            "测试必须显式设置 httpx.post 的返回/异常"
        )
        yield mock_post


# ---------------------------------------------------------------------------
# 超时配置与超时重试
# ---------------------------------------------------------------------------


class TestTimeoutAndRetry:
    def test_timeout_is_split_connect_read(self):
        import ask

        assert ask.HTTP_TIMEOUT.connect == 10.0
        assert ask.HTTP_TIMEOUT.read == 60.0
        assert ask.HTTP_TIMEOUT.write == 30.0

    def test_timeout_raises_connection_error_after_retries(self, no_real_network):
        no_real_network.side_effect = [
            __import__("httpx").TimeoutException("read timeout") for _ in range(3)
        ]
        from ask import _call_llm

        with pytest.raises(LLMConnectionError, match="超时"):
            _call_llm("hi")
        assert no_real_network.call_count == 3  # 有限重试：3 次尝试

    def test_timeout_then_success_retries(self, no_real_network):
        import httpx

        no_real_network.side_effect = [
            httpx.TimeoutException("t"),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"
        assert no_real_network.call_count == 2

    def test_connection_error_retried(self, no_real_network):
        import httpx

        no_real_network.side_effect = [
            httpx.ConnectError("no route"),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"

    def test_connect_timeout_retried(self, no_real_network):
        import httpx

        no_real_network.side_effect = [
            httpx.ConnectTimeout("slow"),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"


# ---------------------------------------------------------------------------
# 重试条件：429 / 5xx 重试；4xx 认证与参数错误不重试
# ---------------------------------------------------------------------------


class TestRetryConditions:
    def test_429_retried_then_success(self, no_real_network):
        no_real_network.side_effect = [
            fake_response(429),
            fake_response(429),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"
        assert no_real_network.call_count == 3

    def test_500_retried_then_success(self, no_real_network):
        no_real_network.side_effect = [
            fake_response(500),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"

    def test_504_retried_then_success(self, no_real_network):
        no_real_network.side_effect = [
            fake_response(504),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm

        assert _call_llm("hi") == "ok"

    def test_429_exhausted_raises_rate_limit(self, no_real_network):
        no_real_network.side_effect = [
            fake_response(429),
            fake_response(429),
            fake_response(429),
        ]
        from ask import _call_llm

        with pytest.raises(LLMRateLimitError):
            _call_llm("hi")
        assert no_real_network.call_count == 3  # 有限重试，不无限循环

    def test_401_not_retried(self, no_real_network):
        no_real_network.side_effect = [fake_response(401)]
        from ask import _call_llm

        with pytest.raises(LLMAuthError):
            _call_llm("hi")
        assert no_real_network.call_count == 1  # 认证错误不重试

    def test_403_not_retried(self, no_real_network):
        no_real_network.side_effect = [fake_response(403)]
        from ask import _call_llm

        with pytest.raises(LLMAuthError):
            _call_llm("hi")
        assert no_real_network.call_count == 1

    def test_400_not_retried(self, no_real_network):
        no_real_network.side_effect = [fake_response(400, {"error": "bad"})]
        from ask import _call_llm

        with pytest.raises(LLMServerError):
            _call_llm("hi")
        assert no_real_network.call_count == 1  # 参数错误不重试

    def test_422_not_retried(self, no_real_network):
        no_real_network.side_effect = [fake_response(422)]
        from ask import _call_llm

        with pytest.raises(LLMServerError):
            _call_llm("hi")
        assert no_real_network.call_count == 1

    def test_sleep_used_between_retries(self, no_real_network):
        no_real_network.side_effect = [
            fake_response(429),
            fake_response(200, OK_BODY),
        ]
        from ask import _call_llm, time as ask_time

        with patch.object(ask_time, "sleep") as mock_sleep:
            assert _call_llm("hi") == "ok"
        mock_sleep.assert_called_once()  # 重试前退避等待


# ---------------------------------------------------------------------------
# 敏感信息脱敏
# ---------------------------------------------------------------------------


class TestSensitiveData:
    def test_auth_error_message_contains_no_key(self, no_real_network):
        no_real_network.side_effect = [fake_response(401)]
        from ask import _call_llm

        with pytest.raises(LLMAuthError) as exc:
            _call_llm("hi")
        assert "sk-" not in str(exc.value)
        assert "Bearer" not in str(exc.value)

    def test_ask_logs_nothing_sensitive(self, no_real_network, caplog):
        """DeepSeek 调用不产生含 Authorization/key 的日志。"""
        no_real_network.side_effect = [fake_response(401)]
        from ask import _call_llm

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMAuthError):
                _call_llm("hi")
        assert "Bearer" not in caplog.text
        assert "Authorization" not in caplog.text

    def test_request_headers_contain_auth_but_body_has_no_key(self, no_real_network):
        """请求头带 Authorization 是必要的；消息体/日志不带 key。"""
        no_real_network.side_effect = [fake_response(200, OK_BODY)]
        from ask import _call_llm

        _call_llm("question here")
        kwargs = no_real_network.call_args.kwargs
        assert "Authorization" in kwargs["headers"]
        assert "sk-" not in json.dumps(kwargs["json"])  # body 不含 key


# ---------------------------------------------------------------------------
# 上传限制：默认 10MB / 空文件 / 路径穿越 / 415 / 413
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from api import app

    return TestClient(app)


class TestUploadLimits:
    def test_default_max_upload_is_10mb(self):
        import api

        assert api.MAX_UPLOAD_BYTES == 10 * 1024 * 1024

    def test_oversized_file_returns_413(self, client):
        import api

        original = api.MAX_UPLOAD_BYTES
        try:
            api.MAX_UPLOAD_BYTES = 100
            resp = client.post(
                "/documents/upload",
                files={"file": ("big.txt", io.BytesIO(b"x" * 5000), "text/plain")},
            )
            assert resp.status_code == 413
            assert "过大" in resp.json()["detail"]
        finally:
            api.MAX_UPLOAD_BYTES = original

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert resp.status_code == 400

    def test_path_traversal_returns_400(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("../../etc/passwd.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert resp.status_code == 400

    def test_unsupported_type_returns_415(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("evil.exe", io.BytesIO(b"x"), "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_success_does_not_leave_temp_files(self, client, tmp_path):
        """上传成功后临时目录被清理（不残留）。"""
        fake_dir = tmp_path / "upload_tmp"
        fake_dir.mkdir()
        with (
            patch("tempfile.mkdtemp", return_value=str(fake_dir)),
            patch("api.add_chunks", return_value=1),
            patch("api.delete_source"),
            patch("api.load_document_paragraphs") as mock_paras,
        ):
            from chunker import Chunk

            mock_paras.return_value = [Chunk(text="t", source="n.txt", page=1)]
            resp = client.post(
                "/documents/upload",
                files={"file": ("note.txt", io.BytesIO(b"hello " * 20), "text/plain")},
            )
            assert resp.status_code == 200
        assert not fake_dir.exists()  # 上传后的 finally 清理已删除临时目录


# ---------------------------------------------------------------------------
# request_id 与结构化日志
# ---------------------------------------------------------------------------


class TestRequestId:
    def test_response_has_request_id_header(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        rid = resp.headers.get("X-Request-ID")
        assert rid and len(rid) >= 8

    def test_request_ids_differ_across_requests(self, client):
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]

    def test_structured_log_contains_metadata_not_body(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="ai_server"):
            client.get("/health")
        assert caplog.records
        record = caplog.records[-1]
        assert record.getMessage()
        msg = record.getMessage()
        assert "request_id=" in msg
        assert "method=GET" in msg
        assert "path=/health" in msg
        assert "status=200" in msg
        assert "duration_ms=" in msg

    def test_upload_log_contains_no_document_content(self, client, caplog):
        secret_content = "TOP_SECRET_DOCUMENT_BODY_XYZ"
        with (
            patch("api.add_chunks", return_value=1),
            patch("api.delete_source"),
            patch("api.load_document_paragraphs") as mock_paras,
        ):
            from chunker import Chunk

            mock_paras.return_value = [Chunk(text="t", source="n.txt", page=1)]
            with caplog.at_level(logging.INFO, logger="ai_server"):
                resp = client.post(
                    "/documents/upload",
                    files={
                        "file": ("note.txt", io.BytesIO(secret_content.encode()), "text/plain")
                    },
                )
                assert resp.status_code == 200
        assert secret_content not in caplog.text  # 不记录完整文档内容


# ---------------------------------------------------------------------------
# 未知异常不暴露堆栈与本地路径
# ---------------------------------------------------------------------------


class TestUnknownErrors:
    @pytest.fixture
    def error_client(self):
        """500 路径测试：不让 TestClient 把服务器异常直接抛给测试。"""
        from api import app

        return TestClient(app, raise_server_exceptions=False)

    def test_internal_error_hides_path_and_traceback(self, error_client):
        with patch("api.stats", side_effect=RuntimeError(
            r"boom at C:\Users\secret\folder\file.py line 42"
        )):
            resp = error_client.get("/health")
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "secret" not in detail
        assert "C:" not in detail
        assert "Traceback" not in detail
        assert detail == "服务器内部错误"

    def test_internal_error_log_has_request_id(self, error_client, caplog):
        with (
            patch("api.stats", side_effect=RuntimeError("boom")),
            caplog.at_level(logging.ERROR, logger="ai_server"),
        ):
            resp = error_client.get("/health")
        assert resp.status_code == 500
        assert any("未处理异常" in r.getMessage() for r in caplog.records)
        assert any("request_id=" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 上游异常语义保持（502/503）
# ---------------------------------------------------------------------------


class TestUpstreamSemantics:
    def test_auth_502(self, client):
        with patch("api.ask", side_effect=LLMAuthError("bad key")):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 502

    def test_rate_limit_503(self, client):
        with patch("api.ask", side_effect=LLMRateLimitError("too many")):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 503

    def test_server_error_502(self, client):
        with patch("api.ask", side_effect=LLMServerError("upstream down")):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 502

    def test_connection_error_502(self, client):
        with patch("api.ask", side_effect=LLMConnectionError("timeout")):
            resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 502
