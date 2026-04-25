"""Tests for OpenAICompatibleProvider._create_stream retry/raise behavior.

Validates the narrowed exception-handling in providers/openai_compat.py:
- httpx.HTTPError / openai.APIError / ProviderError trigger optional retry path.
- Other exception types bubble up unchanged (no retry attempt).
"""

from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from providers.exceptions import APIError, ProviderError


@pytest.mark.asyncio
class TestCreateStreamRetry:
    """_create_stream should retry only when _get_retry_request_body returns a body."""

    async def test_no_retry_on_success(self, nim_provider):
        sentinel_stream = object()
        nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
            return_value=sentinel_stream
        )

        body = {"model": "x", "messages": []}
        stream, used_body = await nim_provider._create_stream(body)

        assert stream is sentinel_stream
        assert used_body is body
        assert nim_provider._global_rate_limiter.execute_with_retry.await_count == 1

    async def test_httpx_error_with_retry_body_retries(self, nim_provider):
        retry_body = {"model": "x", "messages": [], "retried": True}
        sentinel_stream = object()

        with patch.object(
            nim_provider, "_get_retry_request_body", return_value=retry_body
        ):
            nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
                side_effect=[
                    httpx.HTTPError("405 Method Not Allowed"),
                    sentinel_stream,
                ]
            )

            stream, used_body = await nim_provider._create_stream({"model": "x"})
            assert stream is sentinel_stream
            assert used_body is retry_body
            assert nim_provider._global_rate_limiter.execute_with_retry.await_count == 2

    async def test_provider_error_with_no_retry_body_raises(self, nim_provider):
        with patch.object(nim_provider, "_get_retry_request_body", return_value=None):
            err = ProviderError("upstream broken", status_code=500)
            nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
                side_effect=err
            )

            with pytest.raises(ProviderError):
                await nim_provider._create_stream({"model": "x"})
            # Only one call: no retry attempted.
            assert nim_provider._global_rate_limiter.execute_with_retry.await_count == 1

    async def test_api_error_caught_in_narrow_except(self, nim_provider):
        """providers.exceptions.APIError extends ProviderError -> caught."""
        with patch.object(nim_provider, "_get_retry_request_body", return_value=None):
            err = APIError("boom", status_code=502)
            nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
                side_effect=err
            )

            with pytest.raises(APIError):
                await nim_provider._create_stream({"model": "x"})

    async def test_openai_api_error_caught(self, nim_provider):
        """openai.APIError instances flow through the narrowed except clause."""
        retry_body = {"model": "x", "retried": True}
        sentinel_stream = object()
        with patch.object(
            nim_provider, "_get_retry_request_body", return_value=retry_body
        ):
            err = openai.APIError(
                message="bad", request=httpx.Request("POST", "http://x"), body=None
            )
            nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
                side_effect=[err, sentinel_stream]
            )

            stream, used_body = await nim_provider._create_stream({"model": "x"})
            assert stream is sentinel_stream
            assert used_body is retry_body

    async def test_unrelated_exception_bypasses_narrow_except(self, nim_provider):
        """A non-HTTP/non-Provider exception must propagate without retry attempt."""
        with patch.object(nim_provider, "_get_retry_request_body") as mock_retry:
            nim_provider._global_rate_limiter.execute_with_retry = AsyncMock(
                side_effect=RuntimeError("unexpected")
            )

            with pytest.raises(RuntimeError):
                await nim_provider._create_stream({"model": "x"})
            # _get_retry_request_body must NOT be called for unrelated exceptions.
            mock_retry.assert_not_called()
