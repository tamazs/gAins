"""
auth.py — Password hashing and JWT utilities

How JWT works in one paragraph:
  When a user logs in you create a small JSON object (the "payload") containing
  their user_id and an expiry time, then cryptographically sign it with a secret
  key to produce a token string. The token is sent to the frontend and stored
  there (e.g. localStorage). On every subsequent request the frontend sends the
  token in the Authorization header. The backend verifies the signature — if it
  checks out, you know the payload wasn't tampered with and you trust the user_id
  inside it. No session state is stored on the server at all.
"""

import os
import bcrypt
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from tools.mongo_user_store import get_user_by_email

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

# bcrypt is the industry standard for password storage — it is slow by design
# so that brute-force attacks are expensive.

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the password. Store this, never the plain text."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT creation and verification
# ---------------------------------------------------------------------------

ALGORITHM = "HS256"  # HMAC-SHA256 — fast and standard for API tokens
TOKEN_EXPIRE_HOURS = 24


def _secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET must be set in your .env file.")
    return secret


def create_access_token(user_id: str) -> str:
    """
    Build and sign a JWT.
    The payload contains:
      sub  — the subject (who the token belongs to), here the user_id
      exp  — expiry timestamp; jose rejects the token automatically after this
    """
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """
    Verify the token signature and expiry, then return the user_id (sub).
    Raises HTTPException 401 if anything is wrong.
    """
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token: missing sub claim",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI dependency — use this to protect any endpoint
# ---------------------------------------------------------------------------

# OAuth2PasswordBearer tells FastAPI to expect an "Authorization: Bearer <token>"
# header and extract the token string automatically.
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(_oauth2_scheme)) -> str:
    """
    FastAPI dependency that returns the user_id from a valid JWT.

    Usage — add to any endpoint that requires a logged-in user:
        @app.get("/protected")
        def protected(user_id: str = Depends(get_current_user)):
            ...
    """
    return decode_access_token(token)
