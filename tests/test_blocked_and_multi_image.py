"""Plemio fork — blocked generations must explain themselves, and reference images must scale.

Two upstream behaviours these tests pin down:

1. When Gemini refuses a generation (copyrighted character, real-person likeness, …) it
   answers with a candidate whose ``content.parts`` is ``None``. Upstream's
   ``getattr(candidate.content, "parts", [])`` only guards a *missing* attribute, so the
   loop below it blew up with ``TypeError: 'NoneType' object is not iterable`` — an error
   that tells the calling agent nothing and sends it into a retry loop.

2. Reference images were capped at 3 and exposed as three scalar parameters, while Gemini 3
   accepts up to 14. Worse, edit mode was only auto-detected with *exactly one* input image,
   so asking for "keep these faces, add those characters" silently degraded to a from-scratch
   generation.
"""

import os

import pytest
from fastmcp import FastMCP
from fastmcp.utilities.types import Image as MCPImage


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class _FakeInlineData:
    def __init__(self, data: bytes):
        self.data = data


class _FakePart:
    def __init__(self, data: bytes):
        self.inline_data = _FakeInlineData(data)


class _FakeContent:
    """Mirrors google.genai's pydantic Content: ``parts`` exists but may be None."""

    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts, finish_reason=None, finish_message=None, safety_ratings=None):
        self.content = _FakeContent(parts)
        self.finish_reason = finish_reason
        self.finish_message = finish_message
        self.safety_ratings = safety_ratings or []


class _FakePromptFeedback:
    def __init__(self, block_reason=None, block_reason_message=None):
        self.block_reason = block_reason
        self.block_reason_message = block_reason_message


class _FakeResponse:
    def __init__(self, candidates, prompt_feedback=None):
        self.candidates = candidates
        self.prompt_feedback = prompt_feedback


class _FakeEnum:
    """Stand-in for google.genai's FinishReason / BlockedReason enums."""

    def __init__(self, name: str):
        self.name = name
        self.value = name

    def __str__(self) -> str:  # pragma: no cover - defensive
        return f"FinishReason.{self.name}"


def _client() -> "object":
    from nanobanana_mcp_server.services.gemini_client import GeminiClient

    # extract_images / describe_block never touch config or the network.
    return GeminiClient.__new__(GeminiClient)


# --------------------------------------------------------------------------------------
# 1. Blocked generations
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_images_survives_parts_none():
    """A blocked candidate carries ``parts=None`` — that must read as "no images", not a crash."""
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse([_FakeCandidate(parts=None)])

    assert client.extract_images(response) == []


@pytest.mark.unit
def test_extract_images_still_reads_real_parts():
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse([_FakeCandidate(parts=[_FakePart(b"png-bytes")])])

    assert client.extract_images(response) == [b"png-bytes"]


@pytest.mark.unit
def test_describe_block_reports_finish_reason_and_guidance():
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse(
        [_FakeCandidate(parts=None, finish_reason=_FakeEnum("PROHIBITED_CONTENT"))]
    )

    reason = client.describe_block(response)

    assert reason is not None
    assert "PROHIBITED_CONTENT" in reason
    # The message is read by an LLM: it must name a cause and forbid an identical retry.
    assert "copyright" in reason.lower()
    assert "retry" in reason.lower() or "same prompt" in reason.lower()


@pytest.mark.unit
def test_describe_block_prefers_prompt_feedback_when_present():
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse(
        [_FakeCandidate(parts=None)],
        prompt_feedback=_FakePromptFeedback(block_reason=_FakeEnum("SAFETY")),
    )

    reason = client.describe_block(response)

    assert reason is not None and "SAFETY" in reason


@pytest.mark.unit
def test_describe_block_returns_none_when_images_came_back():
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse([_FakeCandidate(parts=[_FakePart(b"x")])])

    assert client.describe_block(response) is None


@pytest.mark.unit
def test_extract_images_or_raise_raises_blocked_with_the_reason():
    import logging

    from nanobanana_mcp_server.core.exceptions import ImageGenerationBlocked

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse(
        [_FakeCandidate(parts=None, finish_reason=_FakeEnum("IMAGE_SAFETY"))]
    )

    with pytest.raises(ImageGenerationBlocked) as excinfo:
        client.extract_images_or_raise(response)

    assert "IMAGE_SAFETY" in str(excinfo.value)


@pytest.mark.unit
def test_extract_images_or_raise_is_transparent_on_success():
    import logging

    client = _client()
    client.logger = logging.getLogger("test")
    response = _FakeResponse([_FakeCandidate(parts=[_FakePart(b"ok")])])

    assert client.extract_images_or_raise(response) == [b"ok"]


@pytest.mark.unit
def test_pro_generate_surfaces_the_block_instead_of_a_typeerror(monkeypatch):
    """End-to-end through the Pro service: a refused generation raises a readable error."""
    from nanobanana_mcp_server.core.exceptions import ImageGenerationBlocked
    from nanobanana_mcp_server.services.gemini_client import GeminiClient
    from nanobanana_mcp_server.services.pro_image_service import ProImageService

    service = ProImageService.__new__(ProImageService)
    import logging

    service.logger = logging.getLogger("test")

    from nanobanana_mcp_server.config.settings import ProImageConfig

    service.config = ProImageConfig()
    service.storage_service = None
    service.progress_tracker = None

    fake_client = GeminiClient.__new__(GeminiClient)
    fake_client.logger = logging.getLogger("test")
    blocked = _FakeResponse(
        [_FakeCandidate(parts=None, finish_reason=_FakeEnum("PROHIBITED_CONTENT"))]
    )
    monkeypatch.setattr(fake_client, "generate_content", lambda *a, **k: blocked, raising=False)
    service.gemini_client = fake_client

    with pytest.raises(ImageGenerationBlocked) as excinfo:
        service.generate_images(prompt="a copyrighted cartoon character", n=1)

    message = str(excinfo.value)
    assert "PROHIBITED_CONTENT" in message
    assert "NoneType" not in message


# --------------------------------------------------------------------------------------
# 2. Multiple reference images
# --------------------------------------------------------------------------------------


class _FakeModelSelector:
    def __init__(self, *, selected_service, selected_tier, model_id: str, name: str):
        self._selected_service = selected_service
        self._selected_tier = selected_tier
        self._model_id = model_id
        self._name = name

    def select_model(self, *args, **kwargs):
        return self._selected_service, self._selected_tier

    def get_model_info(self, tier):
        return {"tier": tier.value, "name": self._name, "model_id": self._model_id, "emoji": "🧪"}


class _FakeProLikeService:
    def __init__(self):
        self.calls = []

    def _result(self):
        return [MCPImage(data=b"thumb", format="jpeg")], [
            {
                "full_path": os.path.join(os.getcwd(), "out.png"),
                "width": 1,
                "height": 1,
                "size_bytes": 4,
            }
        ]

    def generate_images(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return self._result()

    def edit_images(self, **kwargs):
        self.calls.append(("edit", kwargs))
        return self._result()


class _FakeEnhancedImageService:
    def __init__(self):
        self.calls = []

    def edit_image_by_path(self, **kwargs):
        self.calls.append(("path", kwargs))
        return [MCPImage(data=b"thumb", format="jpeg")], [
            {"full_path": os.path.join(os.getcwd(), "out.png"), "width": 1, "height": 1}
        ]


def _get_generate_image_fn():
    from nanobanana_mcp_server.tools.generate_image import register_generate_image_tool

    server = FastMCP("test")
    register_generate_image_tool(server)
    tool = list(server._tool_manager._tools.values())[0]
    return tool.fn


def _images(tmp_path, count: int) -> list[str]:
    paths = []
    for i in range(count):
        p = tmp_path / f"ref{i}.png"
        p.write_bytes(b"fake-png")
        paths.append(str(p))
    return paths


def _wire(monkeypatch, service, tier, enhanced=None):
    selector = _FakeModelSelector(
        selected_service=service, selected_tier=tier, model_id="m", name="Model"
    )
    monkeypatch.setattr("nanobanana_mcp_server.services.get_model_selector", lambda: selector)
    monkeypatch.setattr(
        "nanobanana_mcp_server.tools.generate_image._get_enhanced_image_service",
        lambda: enhanced or _FakeEnhancedImageService(),
    )


@pytest.mark.unit
def test_input_image_paths_array_is_accepted(monkeypatch, tmp_path):
    """The array form mirrors the OpenAI image tool's ``images`` parameter."""
    from nanobanana_mcp_server.config.settings import ModelTier

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    paths = _images(tmp_path, 5)
    result = gen_fn(prompt="fuse these", model_tier="pro", input_image_paths=paths)

    assert result.structured_content["input_image_paths"] == paths
    assert result.structured_content["input_image_count"] == 5


@pytest.mark.unit
def test_several_reference_images_edit_rather_than_regenerate(monkeypatch, tmp_path):
    """>1 image used to fall through to generate mode, losing the source image's subjects."""
    from nanobanana_mcp_server.config.settings import ModelTier

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    paths = _images(tmp_path, 3)
    result = gen_fn(
        prompt="keep the couple's faces, put them in a vineyard with the two characters",
        model_tier="pro",
        input_image_paths=paths,
    )

    assert result.structured_content["mode"] == "edit"
    assert pro.calls and pro.calls[0][0] == "edit"
    # First image is the edit source; the rest travel as additional references.
    kwargs = pro.calls[0][1]
    assert kwargs.get("base_image_b64")
    assert len(kwargs.get("reference_images") or []) == 2


@pytest.mark.unit
def test_legacy_scalar_slots_still_work(monkeypatch, tmp_path):
    from nanobanana_mcp_server.config.settings import ModelTier

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    a, b, c = _images(tmp_path, 3)
    result = gen_fn(
        prompt="fuse", model_tier="pro", input_image_path_1=a, input_image_path_3=c
    )

    assert result.structured_content["input_image_paths"] == [a, c]


@pytest.mark.unit
def test_the_array_wins_over_the_legacy_slots(monkeypatch, tmp_path):
    """Order decides which image is edited, so the two forms must not interleave."""
    from nanobanana_mcp_server.config.settings import ModelTier

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    a, b, c = _images(tmp_path, 3)
    result = gen_fn(
        prompt="fuse",
        model_tier="pro",
        input_image_path_1=a,
        input_image_paths=[c, b, c],  # the repeat collapses
    )

    assert result.structured_content["input_image_paths"] == [c, b]


@pytest.mark.unit
def test_pro_accepts_up_to_fourteen_references(monkeypatch, tmp_path):
    from nanobanana_mcp_server.config.settings import ModelTier

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    paths = _images(tmp_path, 14)
    result = gen_fn(prompt="fuse", model_tier="pro", input_image_paths=paths)

    assert result.structured_content["input_image_count"] == 14


@pytest.mark.unit
def test_over_the_cap_is_a_validation_error_naming_the_limit(monkeypatch, tmp_path):
    from nanobanana_mcp_server.config.settings import ModelTier
    from nanobanana_mcp_server.core.exceptions import ValidationError

    gen_fn = _get_generate_image_fn()
    pro = _FakeProLikeService()
    _wire(monkeypatch, pro, ModelTier.PRO)

    paths = _images(tmp_path, 15)
    with pytest.raises(ValidationError) as excinfo:
        gen_fn(prompt="fuse", model_tier="pro", input_image_paths=paths)

    assert "14" in str(excinfo.value)


@pytest.mark.unit
def test_flash_keeps_the_three_image_cap(monkeypatch, tmp_path):
    """Gemini 2.5 Flash Image never supported more than 3 references."""
    from nanobanana_mcp_server.config.settings import ModelTier
    from nanobanana_mcp_server.core.exceptions import ValidationError

    gen_fn = _get_generate_image_fn()
    enhanced = _FakeEnhancedImageService()
    _wire(monkeypatch, enhanced, ModelTier.FLASH, enhanced=enhanced)

    paths = _images(tmp_path, 4)
    with pytest.raises(ValidationError) as excinfo:
        gen_fn(prompt="fuse", model_tier="flash", input_image_paths=paths)

    assert "3" in str(excinfo.value)
