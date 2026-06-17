from __future__ import annotations

from dataclasses import dataclass

import google.generativeai as genai


@dataclass(frozen=True)
class GeminiAnswer:
    text: str


class GeminiClient:
    def __init__(self, *, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, *, prompt: str) -> GeminiAnswer:
        resp = self.model.generate_content(prompt)
        # Gemini SDK returns different shapes depending on version; this is the common one.
        text = getattr(resp, "text", None) or str(resp)
        return GeminiAnswer(text=text)
