from pydantic import BaseModel, field_validator
import re


class RegisterRequest(BaseModel):
    email: str
    password: str
    username: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v):
        v = v.strip().lower()
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def password_must_be_strong_enough(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("username")
    @classmethod
    def username_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Username cannot be empty")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_normalise(cls, v):
        return v.strip().lower()


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # standard OAuth2 convention
    user_id: str
    username: str
