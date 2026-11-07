"""Language routing logic directing audio to language-specific ASR services."""

from app.core.config import settings

# Supported target languages with deployed ASR services
SUPPORTED_ASR_LANGUAGES = {"en", "te", "hi"}
SOUTH_INDIAN_LANGUAGES = {"te", "ta", "kn", "ml"}


class LanguageRouter:
    """Routes spoken language code to the appropriate IndicConformer or Whisper microservice."""

    def select_asr_endpoint(self, language_code: str) -> tuple[str | None, str]:
        """Map language code to (asr_service_url, normalized_lang).
        
        Returns (None, lang) if the language is unsupported by available ASR models.
        Strictly prevents silent model substitution.
        """
        lang = (language_code or "unknown").lower().strip()

        if lang == "te":
            return settings.asr_te_service_url, "te"
        elif lang == "hi":
            return settings.asr_hi_service_url, "hi"
        elif lang == "en":
            return settings.asr_en_service_url, "en"
        else:
            # Explicitly unsupported language - do NOT silently substitute an incorrect model
            return None, lang


language_router = LanguageRouter()
