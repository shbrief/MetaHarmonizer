"""Stage 4: LLM-based matching using Gemini."""
import os
import json
from typing import List, Tuple, Optional
import google.generativeai as genai
from .base import BaseMatcher
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class GeminiMatcher(BaseMatcher):
    """LLM-based schema matching using Gemini."""
    
    def __init__(self, engine):
        """
        Initialize Gemini matcher.
        
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
        self.model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        
        logger.info("[GeminiMatcher] Initialized with gemini-2.0-flash-exp")
    
    def _build_prompt(self, col: str, sample_values: List[str]) -> str:
        """
        Build prompt for Gemini to match column to standard fields.
        
        Args:
            col: Column name to match
            sample_values: Sample values from the column
            
        Returns:
            Formatted prompt string
        """
        # Get standard field list
        field_list = "\n".join([f"- {f}" for f in self.engine.standard_fields[:50]])  # Limit to avoid token overflow
        
        # Format sample values
        values_str = ", ".join([f'"{v}"' for v in sample_values[:10]])
        
        prompt = f"""You are a clinical data schema mapper. Your task is to map a source column to standard clinical data fields.

Source Column Name: "{col}"
Sample Values: {values_str}

Standard Fields (select from this list):
{field_list}

Instructions:
1. Analyze the column name and sample values
2. Return the top 5 most likely matching standard fields
3. Return your answer as a JSON array with this exact format:
[
  {{"field": "field_name_1", "confidence": 0.95, "reasoning": "brief explanation"}},
  {{"field": "field_name_2", "confidence": 0.80, "reasoning": "brief explanation"}},
  ...
]

IMPORTANT:
- confidence must be a number between 0 and 1
- Only suggest fields from the provided standard fields list
- If no good match exists, return an empty array []
- Return ONLY the JSON array, no other text

JSON Response:"""
        
        return prompt
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """
        Match column using Gemini LLM.
        
        Args:
            col: Column name to match
            
        Returns:
            List of (field_name, score, source) tuples
        """
        try:
            # Get sample values from the column
            sample_values = self.engine.unique_values(col, cap=10)
            
            # Build prompt
            prompt = self._build_prompt(col, sample_values)
            
            # Call Gemini API
            logger.info(f"[Gemini] Calling API for column '{col}'")
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
                logger.warning(f"[Gemini] Invalid response format for '{col}': expected list")
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
                    logger.warning(f"[Gemini] Suggested field '{field}' not in standard fields, skipping")
                    continue
                
                # Validate confidence
                try:
                    score = float(confidence)
                    if not (0 <= score <= 1):
                        logger.warning(f"[Gemini] Invalid confidence {score} for field '{field}', skipping")
                        continue
                except (ValueError, TypeError):
                    logger.warning(f"[Gemini] Non-numeric confidence for field '{field}', skipping")
                    continue
                
                # source = "gemini" to indicate LLM match
                matches.append((field, score, "gemini"))
            
            # Sort by confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"[Gemini] Got {len(matches)} valid matches for '{col}'")
            return matches[:self.engine.top_k]
            
        except json.JSONDecodeError as e:
            logger.error(f"[Gemini] JSON parse error for '{col}': {e}")
            logger.error(f"[Gemini] Response was: {response_text}")
            return []
        except Exception as e:
            logger.error(f"[Gemini] Error matching '{col}': {e}")
            return []