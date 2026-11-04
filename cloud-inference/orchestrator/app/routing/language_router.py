"""Language routing logic directing audio to language-specific ASR services."""

from app.core.config import settings

# Supported target languages and corresponding ASR endpoints
SOUTH_INDIAN_LANGUAGES = {"te", "ta", "kn", "ml"}
NORTH_INDIAN_LANGUAGES = {"hi", "mr", "gu", "pa", "bn", "or", "ur"}


class LanguageRouter:
    """Routes spoken language code to the appropriate IndicConformer microservice."""

    def select_asr_endpoint(self, language_code: str) -> tuple[str, str]:
        """Map language code to (asr_service_url, normalized_lang)."""
        lang = (language_code or "en").lower().strip()

        if lang == "te":
            return settings.asr_te_service_url, "te"
        elif lang == "hi":
            return settings.asr_hi_service_url, "hi"
        elif lang == "en":
            return settings.asr_en_service_url, "en"
        elif lang in SOUTH_INDIAN_LANGUAGES:
            # Fallback to Telugu/Indic endpoint for South Indian cluster until dedicated instances boot
            return settings.asr_te_service_url, lang
        else:
            # General fallback to English
            return settings.asr_en_service_url, "en"


language_router = LanguageRouter()
