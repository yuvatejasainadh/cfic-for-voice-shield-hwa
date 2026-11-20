"""Spoof model selector directing audio to VMC-South or AASIST-L Fallback."""

from app.core.config import settings
from app.routing.language_router import SOUTH_INDIAN_LANGUAGES


class SpoofModelSelector:
    """Selects regional acoustic spoof detector based on detected language."""

    def select_spoof_endpoint(self, language_code: str) -> tuple[str, str]:
        """Map language to (spoof_service_url, model_identifier).
        
        If PellaV2 is configured as the designated acoustic detector,
        all acoustic spoof requests route to PellaV2.
        Otherwise, falls back to regional routing:
        - South Indian languages (te, ta, kn, ml) -> VMC-South (fine-tuned AASIST-L)
        - English & others -> AASIST-L Fallback
        """
        if settings.pellav2_service_url and settings.pellav2_service_url.strip():
            return settings.pellav2_service_url, "pellav2-k12-int8"

        lang = (language_code or "en").lower().strip()
        if lang in SOUTH_INDIAN_LANGUAGES:
            return settings.vmc_south_service_url, "vmc-south"
        return settings.aasist_fallback_service_url, "aasist-fallback"


spoof_selector = SpoofModelSelector()
