"""
Fallback query interface that tries OpenAI first, then falls back to mock.
"""

import logging
from typing import Dict, Any
from sqlalchemy.orm import Session

from .query_interface import QueryInterface
from .mock_query_interface import MockQueryInterface

# Set up logging
logger = logging.getLogger(__name__)

# Create a separate logger for fallback events
fallback_logger = logging.getLogger('fallback_events')
fallback_handler = logging.FileHandler('fallback_events.log')
fallback_handler.setLevel(logging.INFO)
fallback_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fallback_handler.setFormatter(fallback_formatter)
fallback_logger.addHandler(fallback_handler)
fallback_logger.setLevel(logging.INFO)

class FallbackQueryInterface:
    """Query interface that automatically falls back to mock when OpenAI fails."""
    
    def __init__(self):
        self.openai_interface = None
        self.mock_interface = MockQueryInterface()
        self.use_mock = False
        
        try:
            self.openai_interface = QueryInterface()
            fallback_logger.info("OpenAI interface initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI interface initialization failed: {e}")
            fallback_logger.warning(f"OpenAI interface initialization failed: {e}, switching to mock")
            self.use_mock = True
    
    def process_natural_language_query(self, query: str, db: Session) -> Dict[str, Any]:
        """Process query with automatic fallback to mock interface."""
        
        # If we're already using mock, skip OpenAI
        if self.use_mock:
            fallback_logger.info(f"Using mock interface for query: {query[:100]}...")
            return self.mock_interface.process_natural_language_query(query, db)
        
        # Try OpenAI interface first
        try:
            fallback_logger.info(f"Attempting OpenAI interface for query: {query[:100]}...")
            result = self.openai_interface.process_natural_language_query(query, db)
            
            # Check if the result indicates an error (like raw function call)
            if isinstance(result.get('response'), str) and result['response'].startswith('{"function":'):
                # This looks like a failed OpenAI response, fall back to mock
                fallback_logger.warning(f"OpenAI returned raw function call, falling back to mock for query: {query[:100]}...")
                self.use_mock = True
                return self.mock_interface.process_natural_language_query(query, db)
            
            fallback_logger.info(f"OpenAI interface succeeded for query: {query[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI interface failed: {e}")
            fallback_logger.error(f"OpenAI interface failed: {e}, falling back to mock for query: {query[:100]}...")
            self.use_mock = True
            return self.mock_interface.process_natural_language_query(query, db)
