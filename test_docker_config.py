"""
Docker 部署配置的静态检查（不依赖 Docker 守护进程）。

覆盖：Dockerfile 指令与安全要求（slim 镜像 / 非 root / 端口 / 启动命令 /
不复制敏感文件）、.dockerignore 排除清单、docker-compose.yml 结构
（无硬编码 API key / 端口绑定 / 持久卷 / healthcheck / MAX_UPLOAD_BYTES）。

运行：pytest test_docker_config.py -v
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return (ROOT / ".dockerignore").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_uses_official_python_slim(self, dockerfile):
        from_line = next(l for l in dockerfile.splitlines() if l.startswith("FROM "))
        assert "python:" in from_line
        assert "slim" in from_line

    def test_installs_requirements(self, dockerfile):
        assert "COPY requirements.txt" in dockerfile
        assert "pip install" in dockerfile

    def test_copies_only_runtime_sources(self, dockerfile):
        """只复制运行时源码，不复制 .env / 向量库 / 测试 / 评估脚本。"""
        copy_lines = [l for l in dockerfile.splitlines() if l.startswith("COPY ")]
        source_copy = next(
            l for l in copy_lines if "requirements.txt" not in l
        )
        copied = set(source_copy.removeprefix("COPY ").split())
        assert copied == {"api.py", "ask.py", "chunker.py", "store.py", "reranker.py", "./"}
        for forbidden in (".env", "chroma_data", "chroma_data_v2", "test_", "eval_"):
            assert not any(forbidden in c for c in copied), forbidden

    def test_runs_as_non_root(self, dockerfile):
        assert "useradd" in dockerfile
        assert "USER appuser" in dockerfile

    def test_exposes_8000(self, dockerfile):
        assert "EXPOSE 8000" in dockerfile

    def test_uvicorn_command(self, dockerfile):
        # CMD 是 JSON 数组形式：["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
        assert "uvicorn" in dockerfile
        assert "api:app" in dockerfile
        assert '"--host"' in dockerfile and '"0.0.0.0"' in dockerfile
        assert '"--port"' in dockerfile and '"8000"' in dockerfile

    def test_cpu_torch_installed(self, dockerfile):
        """容器内用 CPU 版 torch（避免 2GB+ CUDA 包），与无 GPU 的部署环境匹配。"""
        assert "download.pytorch.org/whl/cpu" in dockerfile

    def test_env_not_copied_into_image(self, dockerfile):
        """没有任何 COPY .env 指令（注释文字不参与判断）。"""
        copy_lines = [l for l in dockerfile.splitlines() if l.startswith("COPY ")]
        assert all(".env" not in l for l in copy_lines)


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


class TestDockerignore:
    REQUIRED = [
        ".env",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".claude",
        "chroma_data",
        "chroma_data_v2",
        "models",
        ".cache",
        "*.log",
        "eval_rewrite_cache.json",
    ]

    def test_covers_all_required_entries(self, dockerignore):
        lines = {l.strip() for l in dockerignore.splitlines() if l.strip() and not l.startswith("#")}
        for entry in self.REQUIRED:
            assert entry in lines, f".dockerignore 缺少：{entry}"

    def test_excludes_vector_stores_and_secrets(self, dockerignore):
        for bad in ("chroma_data", ".env", "eval_rewrite_cache"):
            assert bad in dockerignore


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


class TestCompose:
    def test_yaml_parses(self, compose):
        assert "services" in compose
        assert "ai-knowledge-server" in compose["services"]

    def test_no_hardcoded_api_key(self, compose):
        """API key 只能来自 env_file / 环境变量，不得写死在 yml。"""
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "DEEPSEEK_API_KEY=" not in text.replace("${", "$") or \
            "DEEPSEEK_API_KEY:" not in text
        assert "sk-" not in text.lower()

    def test_env_file_provides_secrets(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        assert ".env" in svc.get("env_file", [])

    def test_port_bound_to_localhost_only(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        ports = svc["ports"]
        assert "127.0.0.1:8000:8000" in ports  # 不发布到公网

    def test_vector_store_persistent_volume(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        vols = svc["volumes"]
        assert any("chroma_data_v2:/app/chroma_data_v2" in v for v in vols)

    def test_healthcheck_hits_health(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        hc = svc["healthcheck"]
        assert "/health" in " ".join(hc["test"])
        assert hc["interval"].endswith("s")

    def test_max_upload_bytes_configured(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        env = svc.get("environment", {})
        assert "MAX_UPLOAD_BYTES" in env

    def test_hf_cache_volume_optional(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        vols = svc["volumes"]
        assert any("hf_cache:/models" in v for v in vols)

    def test_no_tests_or_docs_in_image_build(self, compose):
        svc = compose["services"]["ai-knowledge-server"]
        build = svc.get("build", {})
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "docs" in dockerignore  # docs 不打包进镜像
