from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from sanitizer.policy import get_action
from sanitizer.scrubbers import apply_action

# Presidio's own default recognizers do NOT cover street addresses or
# medical terms out of the box (confirmed empirically before writing this:
# they only caught PERSON and a bare city name as LOCATION) -- both gaps
# closed with custom recognizers below rather than assumed away.
_ADDRESS_PATTERN = Pattern(
    name="street_address",
    regex=(
        r"\d+[A-Za-z]?\s+[A-Za-z0-9.'\s]+?\s+"
        r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|"
        r"Drive|Dr|Court|Ct|Place|Pl|Way)\.?\b"
    ),
    score=0.85,
)

# A curated seed list, not exhaustive -- real medical NER at this
# vocabulary's scale would need a specialized model (e.g. scispaCy),
# beyond this task's scope.
_MEDICAL_TERMS = [
    "diabetes mellitus", "diabetes", "hypertension", "asthma", "depression",
    "anxiety disorder", "epilepsy", "migraine", "arthritis", "leukemia",
    "pneumonia", "tuberculosis", "chronic kidney disease", "copd",
    "myocardial infarction", "alzheimer's disease", "parkinson's disease",
    "hiv", "aids", "cancer",
]

_SUPPORTED_ENTITIES = ["PERSON", "LOCATION", "ADDRESS", "MEDICAL_CONDITION"]


def _build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    analyzer.registry.add_recognizer(
        PatternRecognizer(supported_entity="ADDRESS", patterns=[_ADDRESS_PATTERN])
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(supported_entity="MEDICAL_CONDITION", deny_list=_MEDICAL_TERMS)
    )
    return analyzer


class NerScrubber:
    """Unstructured PII/PHI detection via Presidio + spaCy NER (Phase 2)
    -- catches names, addresses, and medical terms the regex scrubbers
    (structured formats only: email/phone/ssn/credit_card/api_key) never
    attempt to detect. Constructing this loads a spaCy model: build once,
    reuse the instance, don't construct per call.
    """

    def __init__(self) -> None:
        self._analyzer = _build_analyzer()

    def detect(self, text: str) -> list:
        return self._analyzer.analyze(text=text, language="en", entities=_SUPPORTED_ENTITIES)

    def redact(self, text: str, action: str = "mask") -> str:
        results = self.detect(text)
        # replace back-to-front so earlier offsets stay valid as the
        # string shrinks/grows from each replacement
        for result in sorted(results, key=lambda r: r.start, reverse=True):
            replacement = apply_action(text[result.start : result.end], action)
            text = text[: result.start] + replacement + text[result.end :]
        return text

    def redact_with_policy(self, text: str, policy: dict[str, str]) -> str:
        """Like `redact()`, but looks up the action per detected entity
        type via the same policy config regex scrubbing uses (e.g.
        `person`, `address`, `medical_condition`), instead of applying one
        action to every match uniformly."""
        results = self.detect(text)
        for result in sorted(results, key=lambda r: r.start, reverse=True):
            action = get_action(policy, result.entity_type.lower())
            replacement = apply_action(text[result.start : result.end], action)
            text = text[: result.start] + replacement + text[result.end :]
        return text
