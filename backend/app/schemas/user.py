# backend/app/schemas/user.py

from pydantic import BaseModel, EmailStr, ConfigDict, Field, field_validator
from datetime import datetime
import re


class UserBase(BaseModel):
    """Base user schema with shared fields."""
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        examples=["user@example.com"]
    )


class UserCreate(UserBase):
    """Schema for user registration with enhanced validation."""
    password: str = Field(
        ...,
        min_length=6,
        max_length=100,
        description="Password must be at least 6 characters",
        examples=["SecurePassword123"]
    )
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if len(v) > 100:
            raise ValueError('Password must be less than 100 characters')
        # Optional: Check for common weak passwords
        weak_passwords = ['password', '123456', 'qwerty', 'abc123']
        if v.lower() in weak_passwords:
            raise ValueError('Password is too weak. Please choose a stronger password')
        return v


class UserRead(UserBase):
    """Schema for reading user data (no password exposure)."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
