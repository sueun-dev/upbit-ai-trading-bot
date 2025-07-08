"""OpenAI client for trading analysis.

This module provides a standardized interface for OpenAI API
interactions.
"""

import json
from typing import Any, Dict

import openai

# Default configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3


class OpenAIClient:
    """Client for OpenAI API interactions.
    
    This class provides a standardized interface for all OpenAI
    API interactions used in trading analysis.
    
    Attributes:
        client: OpenAI client instance.
        model: Default model to use for completions.
    """

    def __init__(self, api_key: str) -> None:
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key.
            model: OpenAI model to use for completions.
        """
        self.client = openai.OpenAI(api_key=api_key)
    
    # USED
    def analyze_with_prompt(
        self,
        prompt: str,
        system_message: str,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> Dict[str, Any]:
        """Analyze data using OpenAI with a custom prompt.
        
        Args:
            prompt: User prompt for analysis.
            system_message: System message defining AI role.
            temperature: Randomness in AI response (0.0 - 1.0).
            
        Returns:
            Parsed JSON response from OpenAI.
            
        Raises:
            Exception: If analysis fails or response is invalid.
        """
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        return self._parse_response(response)
    
    # USED
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse OpenAI API response.
        
        Args:
            response: Raw API response.
            
        Returns:
            Parsed response as dictionary.
        """
        content = response.choices[0].message.content
        
        # Try to extract JSON from markdown code blocks
        json_content = self._extract_json_from_markdown(content)
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {"response": content}
    
    # USED
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON from markdown code blocks.
        
        Args:
            content: Raw content that may contain JSON in markdown.
            
        Returns:
            Extracted JSON string or original content.
        """
        # Look for ```json ... ``` or ``` ... ``` patterns
        lines = content.strip().split('\n')
        
        # Find start and end of code block
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                if start_idx is None:
                    start_idx = i + 1  # Skip the ``` line
                else:
                    end_idx = i  # Don't include the closing ``` line
                    break
        
        if start_idx is not None and end_idx is not None:
            # Extract content between code blocks
            json_lines = lines[start_idx:end_idx]
            return '\n'.join(json_lines)
        
        # If no code blocks found, return original content
        return content
