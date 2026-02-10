"""Stage 4: LLM-based fallback matching."""
import os
import json
from typing import List, Tuple, Optional
import google.generativeai as genai
from .base import BaseMatcher
from ..config import LLM_MODEL
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class LLMMatcher(BaseMatcher):
    """LLM-based fallback matching using Gemini/Gemma."""
    
    def __init__(self, engine):
        """
        Initialize LLM matcher.
        
        Args:
            engine: Reference to SchemaMapEngine
        """
        super().__init__(engine)
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use model from config
        model_name = LLM_MODEL
        
        # Add 'models/' prefix if not present
        if not model_name.startswith('models/'):
            model_name = f'models/{model_name}'
        
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"[LLM] Initialized with {model_name}")
    
    def _build_prompt(self, col: str, sample_values: Optional[List[str]] = None) -> str:
        """
        Build prompt for LLM to match column to standard fields.
        
        Args:
            col: Column name to match
            sample_values: Optional sample values from the column
            
        Returns:
            Formatted prompt string
        """
        # Get standard field list (limit to avoid token overflow)
        field_list = "\n".join([f"- {f}" for f in self.engine.standard_fields[:100]])
        
        # Format sample values if provided
        values_part = ""
        if sample_values:
            values_str = ", ".join([f'"{v}"' for v in sample_values[:10]])
            values_part = f"\nSample Values: {values_str}"
        
        prompt = f"""You are a clinical data schema mapper. Your task is to map a source column to standard clinical data fields.

Source Column Name: "{col}"{values_part}

Standard Fields (select from this list):
{field_list}

Instructions:
1. Analyze the column name{' and sample values' if sample_values else ''}
2. Return the top {self.engine.top_k} most likely matching standard fields
3. Consider semantic similarity, medical terminology, and common naming conventions
4. Return your answer as a JSON array with this exact format:
[
  {{"field": "field_name_1", "confidence": 0.95, "reasoning": "brief explanation"}},
  {{"field": "field_name_2", "confidence": 0.80, "reasoning": "brief explanation"}},
  ...
]

CRITICAL FORMAT REQUIREMENTS:
- confidence MUST be a decimal number between 0 and 1 (e.g., 0.95, not "95%" or "high" or 95)
- field MUST be exactly as listed in standard fields (copy-paste, don't modify)
- Return PURE JSON only - no markdown code blocks, no explanations
- If no good match exists, return an empty array []

EXAMPLES:
{{"field": "age_at_diagnosis", "confidence": 0.95}}
{{"field": "sex", "confidence": 0.80}}

JSON Response:"""
        
        return prompt
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """
        Match column using LLM.
        
        Args:
            col: Column name to match
            
        Returns:
            List of (field_name, score, source) tuples
        """
        try:
            # Get sample values from the column
            sample_values = self.engine.unique_values(col, cap=10) if hasattr(self.engine, 'unique_values') else None
            
            # Build prompt
            prompt = self._build_prompt(col, sample_values)
            
            # Call LLM API
            logger.info(f"[LLM] Calling API for column '{col}'")
            response = self.model.generate_content(prompt)
            
            # Parse response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            matches_raw = json.loads(response_text)
            
            if not isinstance(matches_raw, list):
                logger.warning(f"[LLM] Invalid response format for '{col}': expected list")
                return []
            
            # Convert to tuple format: (field, score, source)
            matches = []
            for item in matches_raw:
                if not isinstance(item, dict):
                    continue
                
                field = item.get("field", "")
                confidence = item.get("confidence", 0.0)
                
                # Validate field exists in standard fields
                if field not in self.engine.standard_fields:
                    logger.warning(f"[LLM] Suggested field '{field}' not in standard fields, skipping")
                    continue
                
                # Validate confidence
                try:
                    score = float(confidence)
                    if not (0 <= score <= 1):
                        logger.warning(f"[LLM] Invalid confidence {score} for field '{field}', skipping")
                        continue
                except (ValueError, TypeError):
                    logger.warning(f"[LLM] Non-numeric confidence for field '{field}', skipping")
                    continue
                
                # source = "llm" to indicate LLM match
                matches.append((field, score, "llm"))
            
            # Sort by confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"[LLM] Got {len(matches)} valid matches for '{col}'")
            return matches[:self.engine.top_k]
            
        except json.JSONDecodeError as e:
            logger.error(f"[LLM] JSON parse error for '{col}': {e}")
            logger.error(f"[LLM] Response was: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"[LLM] Error matching '{col}': {e}")
            return []