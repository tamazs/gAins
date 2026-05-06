"""
test_user_models.py — Unit tests for models/user_models.py

Covers all field validators on RegisterRequest and LoginRequest.
"""

import pytest
from pydantic import ValidationError

from models.user_models import RegisterRequest, LoginRequest


class TestRegisterRequest:
    def test_valid_request_passes(self):
        req = RegisterRequest(
            email="alice@example.com",
            password="password123",
            username="alice",
        )
        assert req.email == "alice@example.com"
        assert req.username == "alice"

    def test_email_normalized_to_lowercase(self):
        req = RegisterRequest(
            email="User@EXAMPLE.COM",
            password="password123",
            username="alice",
        )
        assert req.email == "user@example.com"

    def test_email_strips_surrounding_whitespace(self):
        req = RegisterRequest(
            email="  alice@example.com  ",
            password="password123",
            username="alice",
        )
        assert req.email == "alice@example.com"

    def test_email_missing_at_raises_validation_error(self):
        with pytest.raises(ValidationError, match="[Ii]nvalid email"):
            RegisterRequest(email="notanemail", password="password123", username="alice")

    def test_email_empty_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="", password="password123", username="alice")

    def test_email_no_domain_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="user@", password="password123", username="alice")

    def test_password_too_short_raises_validation_error(self):
        with pytest.raises(ValidationError, match="8 characters"):
            RegisterRequest(email="alice@example.com", password="short", username="alice")

    def test_password_exactly_7_chars_raises(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="alice@example.com", password="1234567", username="alice")

    def test_password_exactly_8_chars_passes(self):
        req = RegisterRequest(
            email="alice@example.com",
            password="12345678",
            username="alice",
        )
        assert req.password == "12345678"

    def test_username_empty_string_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="alice@example.com", password="password123", username="")

    def test_username_whitespace_only_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RegisterRequest(email="alice@example.com", password="password123", username="   ")

    def test_username_stripped_of_whitespace(self):
        req = RegisterRequest(
            email="alice@example.com",
            password="password123",
            username="  alice  ",
        )
        assert req.username == "alice"


class TestLoginRequest:
    def test_email_normalized_to_lowercase(self):
        req = LoginRequest(email="USER@EXAMPLE.COM", password="pw")
        assert req.email == "user@example.com"

    def test_email_strips_surrounding_whitespace(self):
        req = LoginRequest(email="  user@example.com  ", password="pw")
        assert req.email == "user@example.com"

    def test_password_stored_as_is(self):
        """LoginRequest does not validate password strength — just stores it."""
        req = LoginRequest(email="u@e.com", password="x")
        assert req.password == "x"
