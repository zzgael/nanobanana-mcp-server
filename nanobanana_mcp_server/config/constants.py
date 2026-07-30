"""Application constants."""

# Supported image formats
SUPPORTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"]

# Default aspect ratio options
ASPECT_RATIOS = ["Square image", "Portrait", "Landscape", "16:9", "4:3"]

# File processing constants
THUMBNAIL_SIZE = 256
TEMP_FILE_SUFFIX = ".tmp"

# Reference-image caps, per model tier. Gemini 3 (Pro Image and 3.1 Flash Image / NB2) takes
# up to 14 reference images — the legacy Gemini 2.5 Flash Image path never went past 3.
# https://ai.google.dev/gemini-api/docs/image-generation
MAX_INPUT_IMAGES_BY_TIER = {
    "flash": 3,
    "nb2": 14,
    "pro": 14,
}

# Kept for callers that predate the per-tier caps; equals the highest supported value.
MAX_INPUT_IMAGES = max(MAX_INPUT_IMAGES_BY_TIER.values())

# Image processing defaults
DEFAULT_IMAGE_FORMAT = "png"
THUMBNAIL_FORMAT = "jpeg"

# Template categories
TEMPLATE_CATEGORIES = {
    "photography": "High-quality photographic images",
    "design": "Logos, branding, and graphic design",
    "commercial": "Product shots and commercial photography",
    "illustration": "Artistic illustrations and drawings",
    "editing": "Photo editing and retouching",
    "artistic": "Creative and artistic rendering",
}

# Error messages
ERROR_MESSAGES = {
    "missing_api_key": "GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment",
    "invalid_image_format": "Unsupported image format. Supported types: {types}",
    "image_too_large": "Image size ({size}) exceeds maximum limit ({limit})",
    "validation_error": "Parameter validation failed: {details}",
    "api_error": "Gemini API error: {details}",
    "unknown_error": "An unexpected error occurred: {details}",
}

# Authentication error messages
AUTH_ERROR_MESSAGES = {
    "vertex_ai_project_required": (
        "Vertex AI authentication requires GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT to be set."
    ),
    "api_key_required": (
        "API Key authentication requires GEMINI_API_KEY or GOOGLE_API_KEY to be set."
    ),
    "no_auth_configured": (
        "No valid authentication configuration found.\n"
        "API Key: Set GEMINI_API_KEY or GOOGLE_API_KEY\n"
        "Vertex AI: Set GCP_PROJECT_ID (and optionally GCP_REGION)"
    ),
}
