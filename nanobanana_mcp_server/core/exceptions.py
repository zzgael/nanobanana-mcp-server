"""Custom exceptions for the Nano Banana MCP Server."""


class NanoBananaError(Exception):
    """Base exception class for all Nano Banana errors."""

    pass


class ConfigurationError(NanoBananaError):
    """Raised when there's a configuration issue."""

    pass


class ValidationError(NanoBananaError):
    """Raised when input validation fails."""

    pass


class GeminiAPIError(NanoBananaError):
    """Raised when Gemini API calls fail."""

    pass


class ImageGenerationBlocked(ValidationError):
    """Raised when Gemini refuses to produce an image.

    A refusal is not a server fault and not a transient failure: Gemini answers 200 with a
    candidate carrying no parts and a ``finish_reason`` such as ``PROHIBITED_CONTENT`` or
    ``IMAGE_SAFETY``. The message must therefore say *why* and tell the caller not to replay
    the same prompt, otherwise an agent burns its turns retrying an identical request.

    Subclasses ValidationError so the tool layer treats it as a caller-fixable error rather
    than an internal crash.
    """

    pass


class ImageProcessingError(NanoBananaError):
    """Raised when image processing fails."""

    pass


class FileOperationError(NanoBananaError):
    """Raised when file operations fail."""

    pass


class AuthenticationError(NanoBananaError):
    """Base exception for authentication errors."""

    pass


class ADCConfigurationError(AuthenticationError):
    """Raised when ADC/Vertex AI configuration is invalid."""

    pass
