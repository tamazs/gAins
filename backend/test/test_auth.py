"""
test_auth.py — Unit tests for auth.py

Covers: password hashing, JWT creation, JWT decoding (happy path + every
error branch), and the RuntimeError when JWT_SECRET is missing.
"""

import pytest
from datetime import datetime, timedelta, timezone

from jose import jwt
from fastapi import HTTPException

from auth import hash_password, verify_password, create_access_token, decode_access_token

_ALGORITHM = "HS256"
_SECRET = "test-secret-key-for-unit-tests"


# ── hash_password ─────────────────────────────────────────────────────────────

class TestHashPassword:
    def test_returns_non_empty_string(self):
        result = hash_password("mysecret")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_is_not_plain_text(self):
        assert hash_password("mysecret") != "mysecret"

    def test_is_bcrypt_format(self):
        assert hash_password("mysecret").startswith("$2b$")

    def test_two_hashes_differ(self):
        """bcrypt uses a random salt — the same plain text produces different hashes."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


# ── verify_password ───────────────────────────────────────────────────────────

class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        h = hash_password("correct")
        assert verify_password("correct", h) is True

    def test_wrong_password_returns_false(self):
        h = hash_password("correct")
        assert verify_password("wrong", h) is False

    def test_empty_password_returns_false(self):
        h = hash_password("correct")
        assert verify_password("", h) is False


# ── create_access_token ───────────────────────────────────────────────────────

class TestCreateAccessToken:
    def test_returns_string(self):
        token = create_access_token("user-123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_is_decodable_with_correct_secret(self):
        token = create_access_token("user-123")
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        assert payload is not None

    def test_token_sub_equals_user_id(self):
        token = create_access_token("user-abc")
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        assert payload["sub"] == "user-abc"

    def test_token_has_expiry_claim(self):
        token = create_access_token("user-123")
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        assert "exp" in payload

    def test_missing_jwt_secret_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            create_access_token("user-123")


# ── decode_access_token ───────────────────────────────────────────────────────

class TestDecodeAccessToken:
    def test_valid_token_returns_user_id(self):
        token = create_access_token("user-xyz")
        result = decode_access_token(token)
        assert result == "user-xyz"

    def test_expired_token_raises_401(self):
        payload = {
            "sub": "user-123",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        }
        expired_token = jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(expired_token)
        assert exc_info.value.status_code == 401

    def test_tampered_signature_raises_401(self):
        token = create_access_token("user-123")
        parts = token.split(".")
        tampered = parts[0] + "." + parts[1] + ".invalidsignature"
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(tampered)
        assert exc_info.value.status_code == 401

    def test_token_missing_sub_claim_raises_401(self):
        payload = {"exp": datetime.now(timezone.utc) + timedelta(hours=1)}
        token_no_sub = jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token_no_sub)
        assert exc_info.value.status_code == 401

    def test_token_signed_with_wrong_secret_raises_401(self):
        payload = {
            "sub": "user-123",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        bad_token = jwt.encode(payload, "wrong-secret", algorithm=_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(bad_token)
        assert exc_info.value.status_code == 401

    def test_completely_invalid_string_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            decode_access_token("not.a.token")
        assert exc_info.value.status_code == 401
