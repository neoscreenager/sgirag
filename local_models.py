import base64
import os
import requests
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

_langfuse_tracer = None
_langfuse_tracing_initialized = False


def _get_langfuse_export_endpoint() -> str | None:
    base_url = os.getenv("LANGFUSE_BASE_URL")
    if not base_url:
        return None
    path = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH", "/api/public/otel/v1/traces")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _get_langfuse_headers() -> dict[str, str]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    headers: dict[str, str] = {"x-langfuse-ingestion-version": "4"}
    if public_key and secret_key:
        encoded = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
        headers["x-langfuse-public-key"] = public_key
    return headers


def _initialize_langfuse_tracing() -> bool:
    global _langfuse_tracer, _langfuse_tracing_initialized
    if _langfuse_tracing_initialized:
        return True

    endpoint = _get_langfuse_export_endpoint()
    if not endpoint:
        _langfuse_tracing_initialized = False
        return False

    timeout = float(os.getenv("LANGFUSE_TIMEOUT", "5"))
    headers = _get_langfuse_headers()

    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "local-pdf-rag"}))
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=timeout)
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)
    _langfuse_tracer = trace.get_tracer(__name__)
    _langfuse_tracing_initialized = True
    return True


@contextmanager
def _trace_span(name: str, attributes: dict[str, Any] | None = None, kind: SpanKind = SpanKind.INTERNAL) -> Iterator[None]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        try:
            yield
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


class LocalLMStudioClient:
    """Client to discover and call local LM Studio models."""

    def __init__(self, base_url: str | None = None, timeout: int = 120):
        self.base_url = self._normalize_base_url(
            base_url or os.getenv("LOCAL_LM_BASE_URL") or DEFAULT_BASE_URL
        )
        self.timeout = int(os.getenv("LOCAL_LM_TIMEOUT", str(timeout)))
        self.tracing_enabled = _initialize_langfuse_tracing()
        self.models = self._fetch_models()
        self.llm_model_id = self._choose_llm_model()
        self.embedding_model_id = self._choose_embedding_model()

    def _normalize_base_url(self, url: str) -> str:
        normalized = url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    def _fetch_models(self) -> list[dict]:
        url = f"{self.base_url}/models"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        models = payload.get("data") or payload.get("models") or payload
        if isinstance(models, dict):
            models = models.get("data") or models.get("models") or []

        if not isinstance(models, list):
            raise RuntimeError(f"Unexpected model list format from {url}: {payload}")

        if not models:
            raise RuntimeError(f"No models returned from {url}")

        return models

    def _choose_embedding_model(self) -> str:
        candidates = []
        for model in self.models:
            model_id = str(model.get("id") or model.get("name") or "").lower()
            model_type = str(model.get("type") or "").lower()
            tags = [str(tag).lower() for tag in model.get("tags") or []]
            if "embed" in model_id or "embed" in model_type or "embeddings" in tags:
                candidates.append(model_id)

        if candidates:
            return candidates[0]

        for model in self.models:
            model_id = str(model.get("id") or model.get("name") or "").lower()
            if "text-embedding" in model_id or "embedding" in model_id:
                return model_id

        return str(self.models[0].get("id") or self.models[0].get("name"))

    def _choose_llm_model(self) -> str:
        candidates = []
        for model in self.models:
            model_id = str(model.get("id") or model.get("name") or "").lower()
            model_type = str(model.get("type") or "").lower()
            tags = [str(tag).lower() for tag in model.get("tags") or []]
            if any(text in model_id for text in ["gpt", "llama", "ministral", "gemini", "claude", "mamba"]):
                candidates.append(model_id)
            elif "completion" in model_type or "chat" in model_type:
                candidates.append(model_id)
            elif "chat" in tags or "completion" in tags:
                candidates.append(model_id)

        if candidates:
            return candidates[0]

        return str(self.models[0].get("id") or self.models[0].get("name"))

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        with _trace_span(
            name="local-lmstudio.http.post",
            attributes={
                "http.method": "POST",
                "http.url": url,
                "component": "local-lmstudio",
                "llm.base_url": self.base_url,
            },
            kind=SpanKind.CLIENT,
        ):
            response = requests.post(url, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text or ""
            raise requests.HTTPError(
                f"HTTP {response.status_code} error for {url}: {body}",
                response=response,
            ) from exc
        return response.json()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.embedding_model_id, "input": texts}
        with _trace_span(
            name="local-lmstudio.embedding",
            attributes={
                "component": "local-lmstudio",
                "llm.model": self.embedding_model_id,
                "llm.operation": "embed_texts",
                "http.url": f"{self.base_url}/embeddings",
                "request.text_length": len("\n".join(texts)),
            },
            kind=SpanKind.CLIENT,
        ):
            response = self._post("/embeddings", body)
        embeddings = response.get("data") or []
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], dict) and "embedding" in embeddings[0]:
            return [item["embedding"] for item in embeddings]
        if isinstance(embeddings, list) and all(isinstance(item, list) for item in embeddings):
            return embeddings
        raise RuntimeError(f"Unexpected embeddings response: {response}")

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _truncate_context(self, context: str, max_chars: int = 12000) -> str:
        if len(context) <= max_chars:
            return context
        return "..." + context[-max_chars:].lstrip()

    def generate_answer(self, prompt: str, context: str) -> str:
        context = self._truncate_context(context)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context. Extract the key value pairs from the PDF document and use them to answer the question as accurately as possible. If the answer cannot be found in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
        ]
        payload = {
            "model": self.llm_model_id,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
        }

        with _trace_span(
            name="local-lmstudio.chat_completion",
            attributes={
                "component": "local-lmstudio",
                "llm.model": self.llm_model_id,
                "llm.operation": "generate_answer",
                "http.url": f"{self.base_url}/chat/completions",
                "request.prompt_length": len(prompt),
                "request.context_length": len(context),
            },
            kind=SpanKind.CLIENT,
        ):
            try:
                response = self._post("/chat/completions", payload)
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    return choices[0].get("message", {}).get("content", "").strip()
            except requests.HTTPError as exc:
                response_text = exc.response.text if exc.response is not None else ""
                status_code = exc.response.status_code if exc.response is not None else None
                should_fallback = status_code in (400, 404)
                if should_fallback:
                    fallback = {
                        "model": self.llm_model_id,
                        "prompt": f"Context:\n{context}\n\nQuestion: {prompt}",
                        "max_tokens": 512,
                        "temperature": 0.2,
                    }
                    try:
                        response = self._post("/completions", fallback)
                    except requests.HTTPError as exc2:
                        fallback_text = exc2.response.text if exc2.response is not None else ""
                        lower_text = fallback_text.lower()
                        if "input" in lower_text and "required" in lower_text:
                            fallback_input = {
                                "model": self.llm_model_id,
                                "input": fallback["prompt"],
                                "max_tokens": 512,
                                "temperature": 0.2,
                            }
                            response = self._post("/completions", fallback_input)
                        elif "prompt" in lower_text and "required" in lower_text:
                            response = self._post("/completions", fallback)
                        else:
                            raise RuntimeError(
                                f"Completion fallback failed ({exc2.response.status_code if exc2.response is not None else 'unknown'}): {fallback_text}",
                            ) from exc2
                    choices = response.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        return choices[0].get("text", "").strip()
                raise RuntimeError(
                    f"Chat completion failed ({status_code}): {response_text}",
                ) from exc

        raise RuntimeError(f"Unexpected completion response: {response}")
