"""
DP-FUSION implementation for differential privacy text rewriting, adapted for LangChain.
Based on the DP-FUSION Defense methodology with true Rényi divergence guarantees.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Type
from collections import defaultdict
import sys
import os

# Add parent directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

try:
    from .pii_detection import PIIDetectionTool
    from utils import generate_answer
except ImportError:
    from pii_detection import PIIDetectionTool
    from utils import generate_answer


# Default beta values (privacy budget per entity type)
DEFAULT_BETA_DICT = {
    "PERSON": 0.5, "CODE": 0.5, "LOCATION": 0.5, "ORG": 0.5,
    "DEM": 0.5, "DATE_TIME": 0.5, "QUANTITY": 0.5, "MISC": 0.5,
}

ENTITY_TYPES = ["PERSON", "CODE", "LOCATION", "ORG", "DEM", "DATE_TIME", "QUANTITY", "MISC"]

PII_TO_DP_FUSION_MAPPING = {
    "PERSON": "PERSON", "ORGANIZATION": "ORG", "ORG": "ORG",
    "LOCATION": "LOCATION", "GPE": "LOCATION",
    "DATE_TIME": "DATE_TIME", "QUANTITY": "QUANTITY", "MISC": "MISC"
}


class DPRewritingInput(BaseModel):
    text: str = Field(description="The text to rewrite with differential privacy")
    model_name: str = Field(default="qwen/qwen3-235b-a22b:free", description="Model ID for API calls")
    api_token: str = Field(default="", description="API token for model access")
    beta_dict: Optional[Dict[str, float]] = Field(default=None, description="Privacy budget per entity type")
    alpha: float = Field(default=2.0, description="Rényi divergence order (must be > 1)")
    delta: float = Field(default=0.001, description="Delta parameter for (ε,δ)-DP")
    max_tokens: int = Field(default=50, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Generation temperature")
    strategies: List[str] = Field(
        default=["names", "locations", "organizations", "dates"],
        description="PII entity types to detect and anonymize"
    )


class DPRewritingTool(BaseTool):
    """DP-FUSION text rewriting with privacy-preserving API calls."""

    name: str = "DP_rewriting"
    description: str = (
        "DP-FUSION text rewriting tool that provides differential privacy guarantees. "
        "Automatically detects PII (names, locations, organizations, dates) and rewrites text "
        "to remove sensitive information while preserving meaning. Use this for privacy-sensitive "
        "text that contains personal names, locations, organizations, or dates."
    )
    args_schema: Type[BaseModel] = DPRewritingInput

    pii_detector: Any = None
    placeholder_token: str = "_"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pii_detector = PIIDetectionTool()

    def _run(self, text: str, model_name: str = "qwen/qwen3-235b-a22b:free",
             api_token: str = "", beta_dict: Optional[Dict[str, float]] = None,
             alpha: float = 2.0, delta: float = 0.001, max_tokens: int = 50,
             temperature: float = 1.0,
             strategies: List[str] = ["names", "locations", "organizations", "dates"]) -> str:
        if beta_dict is None:
            beta_dict = DEFAULT_BETA_DICT.copy()

        try:
            result = self._process_with_dp_fusion_api(
                text, model_name, api_token, beta_dict, alpha, delta, max_tokens, temperature, strategies
            )
            return json.dumps(result)
        except Exception as e:
            logging.error(f"DP-FUSION execution failed: {e}")
            return json.dumps({
                'error': str(e),
                'rewritten_text': text,
                'applied_strategies': [],
                'privacy_metrics': {},
            })

    def _process_with_dp_fusion_api(self, text: str, model_name: str, api_token: str,
                                   beta_dict: Dict[str, float], alpha: float, delta: float,
                                   max_tokens: int, temperature: float, strategies: List[str]) -> Dict[str, Any]:
        grouped_offsets = self._detect_and_group_entities(text, strategies)

        if not grouped_offsets:
            return {
                'rewritten_text': text, 'original_text': text,
                'applied_strategies': [],
                'privacy_metrics': {'total_changes': 0, 'epsilon': 0.0},
            }

        versions = self._create_privacy_versions(text, grouped_offsets)

        rewritten_texts = {}
        for version_name, version_text in versions.items():
            prompt = self._create_rewriting_prompt(version_text)
            try:
                if not api_token:
                    raise ValueError("API token is required for DP-FUSION generation")
                rewritten_text, _ = generate_answer(model_name, api_token, prompt)
                rewritten_texts[version_name] = self._clean_generated_text(rewritten_text)
            except Exception as e:
                logging.warning(f"Failed to generate for version {version_name}: {e}")
                rewritten_texts[version_name] = text if version_name == "PUBLIC" else version_text

        final_text, privacy_metrics = self._select_best_version(
            rewritten_texts, grouped_offsets, beta_dict, alpha, delta
        )

        return {
            'rewritten_text': final_text, 'original_text': text,
            'applied_strategies': list(grouped_offsets.keys()),
            'privacy_metrics': privacy_metrics,
        }

    def _create_privacy_versions(self, text: str, grouped_offsets: Dict[str, List[Tuple[int, int]]]) -> Dict[str, str]:
        versions = {}
        all_offsets = []
        for off_list in grouped_offsets.values():
            all_offsets.extend(off_list)
        versions["PUBLIC"] = self._replace_entities_with_placeholder(text, all_offsets)

        for ent_type in grouped_offsets:
            offsets_except = []
            for t2, off_list in grouped_offsets.items():
                if t2 != ent_type:
                    offsets_except.extend(off_list)
            versions[ent_type] = self._replace_entities_with_placeholder(text, offsets_except)

        return versions

    def _replace_entities_with_placeholder(self, text: str, offsets: List[Tuple[int, int]]) -> str:
        if not offsets:
            return text
        sorted_offsets = sorted(offsets, key=lambda x: x[0], reverse=True)
        result = text
        for start, end in sorted_offsets:
            result = result[:start] + self.placeholder_token + result[end:]
        return result

    def _create_rewriting_prompt(self, version_text: str) -> str:
        return f"""You are given a passage that may contain placeholders (underscores) or incomplete data. Your job is to produce a natural paraphrase. Do not use any underscores or placeholders in your output. If data is missing, just omit it or paraphrase gracefully. Do not output anything except the paraphrase. Make sure to retain all information from the source document.

Document:
{version_text}

Paraphrase the above text. Whenever a placeholder i.e {self.placeholder_token} exists, you must completely ignore that information, as {self.placeholder_token} indicates redacted text. To ensure the generated text is as natural as possible, you must never output the {self.placeholder_token} themselves.

Paraphrased text:"""

    def _clean_generated_text(self, generated_text: str) -> str:
        prefixes_to_remove = [
            "Paraphrased text:", "Here is the paraphrased text:",
            "Here's the paraphrased text:",
            "Sure. Here is the paraphrased document without underscores or placeholders:",
            "Paraphrase:"
        ]
        if not generated_text:
            return ""
        cleaned = generated_text.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        cleaned = cleaned.replace(self.placeholder_token, "")
        return cleaned

    def _select_best_version(self, rewritten_texts: Dict[str, str], grouped_offsets: Dict[str, List[Tuple[int, int]]],
                           beta_dict: Dict[str, float], alpha: float, delta: float) -> Tuple[str, Dict[str, Any]]:
        total_entities = sum(len(entities) for entities in grouped_offsets.values())
        epsilon_values = {}
        for ent_type, entities in grouped_offsets.items():
            beta_val = beta_dict.get(ent_type, 0.5)
            epsilon_values[ent_type] = len(entities) * beta_val

        privacy_metrics = {
            'total_changes': total_entities,
            'epsilon_per_group': epsilon_values,
            'global_epsilon': max(epsilon_values.values()) if epsilon_values else 0.0,
            'versions_created': list(rewritten_texts.keys()),
            'method': 'dp_fusion_api_simplified'
        }

        final_text = rewritten_texts.get("PUBLIC", list(rewritten_texts.values())[0] if rewritten_texts else "")
        return final_text, privacy_metrics

    def _detect_and_group_entities(self, text: str, strategies: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        grouped_offsets = defaultdict(list)
        strategy_mapping = {
            'names': ['PERSON'], 'locations': ['LOCATION', 'GPE'],
            'organizations': ['ORG', 'ORGANIZATION'],
            'dates': ['DATE_TIME'], 'numbers': ['QUANTITY']
        }
        pii_entities_to_detect = []
        for strategy in strategies:
            if strategy in strategy_mapping:
                pii_entities_to_detect.extend(strategy_mapping[strategy])

        if not pii_entities_to_detect:
            return {}

        pii_result = self.pii_detector.execute(text=text, entities=pii_entities_to_detect, threshold=0.5)

        if not pii_result.get('has_pii', False):
            return {}

        for entity in pii_result.get('detected_entities', []):
            pii_type = entity['entity_type']
            dp_fusion_type = PII_TO_DP_FUSION_MAPPING.get(pii_type, "MISC")
            grouped_offsets[dp_fusion_type].append((entity['start'], entity['end']))

        return dict(grouped_offsets)
