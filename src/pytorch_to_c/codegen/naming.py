"""Backend-neutral identifier sanitization for code generators."""


def sanitize_name(name: str) -> str:
    """Sanitize a name to be a valid C/Python identifier."""
    sanitized = name.replace(".", "_").replace("-", "_").replace(" ", "_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized
