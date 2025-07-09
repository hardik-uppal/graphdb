"""
Fallback query interface that tries OpenAI first, then falls back to mock.
"""

from typing import Dict, Any
from sqlalchemy.orm import Session

from .query_interface import QueryInterface
from .mock_query_interface import MockQueryInterface

class FallbackQueryInterface:
    """Query interface that automatically falls back to mock when OpenAI fails."""
    
    def __init__(self):
        self.openai_interface = None
        self.mock_interface = MockQueryInterface()
        self.use_mock = False
        
        try:
            self.openai_interface = QueryInterface()
        except Exception as e:
            print(f"OpenAI interface initialization failed: {e}")
            self.use_mock = True
    
    def process_natural_language_query(self, query: str, db: Session) -> Dict[str, Any]:
        """Process query with automatic fallback to mock interface."""
        
        # If we're already using mock, skip OpenAI
        if self.use_mock:
            return self.mock_interface.process_natural_language_query(query, db)
        
        # Try OpenAI interface first
        try:
            result = self.openai_interface.process_natural_language_query(query, db)
            
            # Check if the result indicates an error (like raw function call)
            if isinstance(result.get('response'), str) and result['response'].startswith('{"function":'):
                # This looks like a failed OpenAI response, fall back to mock
                print("OpenAI returned raw function call, falling back to mock interface")
                self.use_mock = True
                return self.mock_interface.process_natural_language_query(query, db)
            
            return result
            
        except Exception as e:
            print(f"OpenAI interface failed: {e}")
            print("Falling back to mock interface")
            self.use_mock = True
            return self.mock_interface.process_natural_language_query(query, db)
