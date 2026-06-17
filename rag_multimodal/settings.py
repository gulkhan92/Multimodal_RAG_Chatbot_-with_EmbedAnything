from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    quadrant_url: str
    quadrant_path: str
    quadrant_api_key: str
    embedanything_text_model: str
    embedanything_image_model: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    @staticmethod
    def from_env() -> "Settings":
        load_dotenv()

        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        secret_key = os.getenv("SECRET_KEY", "default_secret_for_dev_only").strip()

        quadrant_url = os.getenv("QUADRANT_URL", "").strip()
        quadrant_path = os.getenv("QUADRANT_PATH", "").strip()
        quadrant_api_key = os.getenv("QUADRANT_API_KEY", "").strip()

        embedanything_text_model = os.getenv("EMBEDANYTHING_TEXT_MODEL", "").strip()
        embedanything_image_model = os.getenv("EMBEDANYTHING_IMAGE_MODEL", "").strip()

        if not gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")

        if not quadrant_url and not quadrant_path:
            raise RuntimeError("Missing QUADRANT_URL or QUADRANT_PATH in environment/.env")

        # embed model can be empty depending on EmbedAnything defaults
        return Settings(
            gemini_api_key=gemini_api_key,
            quadrant_url=quadrant_url,
            quadrant_path=quadrant_path,
            quadrant_api_key=quadrant_api_key,
            embedanything_text_model=embedanything_text_model,
            embedanything_image_model=embedanything_image_model,
            secret_key=secret_key
        )
