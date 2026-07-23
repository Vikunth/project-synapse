from __future__ import annotations

from fastapi.responses import JSONResponse
from pydantic import BaseModel


class SynapseError(BaseModel):
    """Structured error model for Synapse."""

    message: str
    type: str
    code: int


def synapse_error_response(
    status_code: int, message: str, error_type: str = "proxy_error"
) -> JSONResponse:
    """
    Create a standard Synapse structured error response.

    Args:
        status_code: HTTP status code.
        message: Error description.
        error_type: Type of the error.

    Returns:
        JSONResponse: The formatted error response.
    """
    error = SynapseError(message=message, type=error_type, code=status_code)
    return JSONResponse(status_code=status_code, content=error.model_dump())


def upstream_unreachable_response(detail: str = "") -> JSONResponse:
    """
    Create a 502 Bad Gateway response when upstream is unreachable.

    Args:
        detail: Additional error details.

    Returns:
        JSONResponse: The formatted error response.
    """
    msg = "Upstream Ollama instance unreachable"
    if detail:
        msg = f"{msg}: {detail}"
    return synapse_error_response(status_code=502, message=msg, error_type="upstream_error")


def upstream_timeout_response() -> JSONResponse:
    """
    Create a 504 Gateway Timeout response when upstream times out.

    Returns:
        JSONResponse: The formatted error response.
    """
    return synapse_error_response(
        status_code=504,
        message="Upstream Ollama instance timed out",
        error_type="upstream_timeout",
    )


def openai_error_response(
    status_code: int, message: str, error_type: str = "server_error"
) -> JSONResponse:
    """
    Create an OpenAI-compatible error response.

    Args:
        status_code: HTTP status code.
        message: Error description.
        error_type: Type of the error.

    Returns:
        JSONResponse: The formatted OpenAI-compatible error response.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
            }
        },
    )
