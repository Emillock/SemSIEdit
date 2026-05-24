"""
PII Detection tool adapted for LangChain.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
except ImportError as e:
    logging.warning(f"Presidio not available: {e}. PII detection will be limited.")
    AnalyzerEngine = None
    AnonymizerEngine = None
    NlpEngineProvider = None


class PIIDetectionInput(BaseModel):
    text: str = Field(description="The text to analyze for PII")
    anonymize: bool = Field(default=False, description="Whether to return anonymized text")
    threshold: float = Field(default=0.5, description="Confidence threshold 0.0-1.0")
    entities: Optional[List[str]] = Field(default=None, description="Specific entity types to detect")
    language: str = Field(default="en", description="Language code")


class PIIDetectionTool(BaseTool):
    """Tool for detecting and anonymizing PII."""

    name: str = "PII_Detection"
    description: str = "Detects personally identifiable information (PII) in text. Returns detected entities with types, positions, and confidence scores."
    args_schema: Type[BaseModel] = PIIDetectionInput

    analyzer: Any = None
    anonymizer: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if AnalyzerEngine is not None:
            try:
                self._initialize_engines()
            except Exception as e:
                logging.warning(f"Failed to initialize Presidio engines: {e}")

    def _initialize_engines(self):
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        try:
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()
        except Exception as e:
            logging.warning(f"Failed to create custom NLP engine, using defaults: {e}")
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

    def _run(self, text: str, anonymize: bool = False, threshold: float = 0.5,
             entities: Optional[List[str]] = None, language: str = "en") -> str:
        if self.analyzer is None:
            result = self._fallback_detection(text, anonymize=anonymize)
        else:
            result = self._presidio_detection(text, entities, threshold, anonymize, language)
        return json.dumps(result)

    def _presidio_detection(self, text: str, entities, threshold, anonymize, language) -> Dict[str, Any]:
        try:
            results = self.analyzer.analyze(
                text=text, entities=entities, language=language, score_threshold=threshold
            )
            detected_entities = [
                {
                    'entity_type': r.entity_type,
                    'start': r.start,
                    'end': r.end,
                    'confidence': r.score,
                    'text': text[r.start:r.end]
                }
                for r in results
            ]
            response = {
                'detected_entities': detected_entities,
                'has_pii': len(detected_entities) > 0
            }
            if anonymize and self.anonymizer is not None:
                anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
                response['anonymized_text'] = anonymized_result.text
            return response
        except Exception as e:
            logging.error(f"Error during PII detection: {e}")
            return {'error': str(e), 'detected_entities': [], 'has_pii': False}

    def _fallback_detection(self, text: str, **kwargs) -> Dict[str, Any]:
        import re
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        detected_entities = []
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                detected_entities.append({
                    'entity_type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'text': match.group()
                })
        response = {'detected_entities': detected_entities, 'has_pii': len(detected_entities) > 0}
        if kwargs.get('anonymize', False):
            anonymized_text = text
            for entity_type, pattern in patterns.items():
                anonymized_text = re.sub(pattern, f'[{entity_type}]', anonymized_text)
            response['anonymized_text'] = anonymized_text
        return response

    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """Legacy execute interface for backward compatibility with DPRewriter."""
        if self.analyzer is None:
            return self._fallback_detection(text, **kwargs)
        return self._presidio_detection(
            text,
            kwargs.get('entities'),
            kwargs.get('threshold', 0.5),
            kwargs.get('anonymize', False),
            kwargs.get('language', 'en')
        )
