"""LLM provider 抽象：Claude（官方 anthropic SDK）/ OpenAI 兼容端点 / 文件交换（人或外部代理充当 LLM）/ 规则 mock。

所有 provider 实现同一个接口：
    chat(messages: list[Msg]) -> str
Msg = {"role": "user"|"assistant", "text": str, "images": [Path, ...]}
system 提示词单独传入构造函数。密钥只从环境变量读，绝不写进代码。

环境变量：
  ANTHROPIC_API_KEY（或 `ant auth login` 的本地凭据）  P6_ANTHROPIC_MODEL（默认 claude-opus-5-5）
  P6_ANTHROPIC_EFFORT（默认 high）  P6_ANTHROPIC_FALLBACKS（默认 1，=0 关闭服务端拒答回退）
  P6_OPENAI_BASE_URL / P6_OPENAI_API_KEY / P6_OPENAI_MODEL     —— 任意 OpenAI 兼容端点：
      OpenAI 官方、阿里云百炼 DashScope 兼容模式、MiniMax 开放平台、本地 vLLM / Ollama / LM Studio
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Protocol

Msg = dict


class Provider(Protocol):
    name: str

    def chat(self, messages: list[Msg]) -> str: ...


def _b64(p: Path) -> str:
    return base64.standard_b64encode(Path(p).read_bytes()).decode("utf-8")


# --------------------------------------------------------------------------- Claude
class AnthropicProvider:
    """Claude Messages API（官方 Python SDK `anthropic`）。流式 + 自适应思考。"""

    def __init__(self, system: str, model: str | None = None, effort: str | None = None, max_tokens: int = 32000):
        import anthropic  # 延迟导入：没装 SDK 也不影响其他 provider

        self.client = anthropic.Anthropic()  # 凭据：ANTHROPIC_API_KEY 或 ant auth login 的 profile
        self.system = system
        self.model = model or os.environ.get("P6_ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = effort or os.environ.get("P6_ANTHROPIC_EFFORT", "high")
        self.max_tokens = max_tokens
        self.fallbacks = os.environ.get("P6_ANTHROPIC_FALLBACKS", "1") != "0"
        self.name = f"anthropic:{self.model}"
        self.usage: list[dict] = []

    def _content(self, m: Msg) -> list[dict]:
        c = []
        for p in m.get("images", []):
            c.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(p)}})
        c.append({"type": "text", "text": m["text"]})
        return c

    def chat(self, messages: list[Msg]) -> str:
        kw = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": m["role"], "content": self._content(m) if m["role"] == "user" else m["text"]}
                      for m in messages],
        )
        if self.fallbacks:
            # 服务端拒答回退（按拒答类别自动路由）；在 Bedrock/Vertex 上不可用，设 P6_ANTHROPIC_FALLBACKS=0
            with self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                  **kw) as s:
                resp = s.get_final_message()
        else:
            with self.client.messages.stream(**kw) as s:
                resp = s.get_final_message()
        self.usage.append({"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                           "stop_reason": resp.stop_reason})
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"模型拒答: {getattr(resp, 'stop_details', None)}")
        return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- OpenAI 兼容
class OpenAICompatProvider:
    """/v1/chat/completions 兼容端点（图片用 data URI）。适配 DashScope(Qwen-VL)、MiniMax、vLLM、Ollama 等。"""

    def __init__(self, system: str, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, max_tokens: int = 16000, timeout: int = 600):
        import requests  # noqa

        self.base = (base_url or os.environ.get("P6_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.key = api_key or os.environ.get("P6_OPENAI_API_KEY", "")
        self.model = model or os.environ.get("P6_OPENAI_MODEL", "")
        if not self.model:
            raise ValueError("需要设置 P6_OPENAI_MODEL（端点上的模型名）")
        self.system = system
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = f"openai-compat:{self.model}@{self.base}"

    def chat(self, messages: list[Msg]) -> str:
        import requests

        msgs = [{"role": "system", "content": self.system}]
        for m in messages:
            if m["role"] == "user":
                c = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}}
                     for p in m.get("images", [])]
                c.append({"type": "text", "text": m["text"]})
                msgs.append({"role": "user", "content": c})
            else:
                msgs.append({"role": "assistant", "content": m["text"]})
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        r = requests.post(f"{self.base}/chat/completions", headers=h, timeout=self.timeout,
                          json={"model": self.model, "messages": msgs, "max_tokens": self.max_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- 文件交换
class FileExchangeProvider:
    """把请求写成 request_NN.md（附图片路径），等待同目录出现 response_NN.md。

    用途：没有 API key 时，让人或外部代理（例如 Claude Code 会话本身）充当 LLM，闭环其余部分全自动。
    """

    def __init__(self, system: str, exchange_dir: Path, poll: float = 3.0, timeout: float = 3600, label: str = "file"):
        self.dir = Path(exchange_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "system.md").write_text(system, encoding="utf-8")
        self.poll, self.timeout = poll, timeout
        self.n = 0
        self.name = f"file-exchange:{label}"

    def chat(self, messages: list[Msg]) -> str:
        self.n += 1
        last = messages[-1]
        req = self.dir / f"request_{self.n:02d}.md"
        imgs = "\n".join(f"- {Path(p).resolve()}" for p in last.get("images", []))
        req.write_text(f"{last['text']}\n\n## 附图\n{imgs}\n", encoding="utf-8")
        resp = self.dir / f"response_{self.n:02d}.md"
        t0 = time.time()
        while not resp.exists():
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"等待 {resp} 超时")
            time.sleep(self.poll)
        time.sleep(0.5)
        return resp.read_text(encoding="utf-8")


def make_provider(kind: str, system: str, **kw) -> Provider:
    from .mock import MockProvider

    if kind == "anthropic":
        return AnthropicProvider(system, **kw)
    if kind == "openai":
        return OpenAICompatProvider(system, **kw)
    if kind == "file":
        return FileExchangeProvider(system, **kw)
    if kind == "mock":
        return MockProvider(system, **kw)
    raise ValueError(kind)


def available_keys() -> dict[str, bool]:
    return {k: bool(os.environ.get(k)) for k in
            ["ANTHROPIC_API_KEY", "P6_OPENAI_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "MINIMAX_API_KEY"]}


if __name__ == "__main__":
    print(json.dumps(available_keys(), indent=2))
