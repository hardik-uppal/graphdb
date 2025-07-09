#!/usr/bin/env python3
"""
Test the updated OpenAI query interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db
from src.query_interface import QueryInterface
from src.fallback_query_interface import FallbackQueryInterface

def test_query_interface():
    """Test the query interface with a simple question."""
    db = next(get_db())
    
    try:
        # Test the updated OpenAI interface
        print("Testing updated OpenAI query interface...")
        interface = QueryInterface()
        
        query = "What did I spend the most money on?"
        print(f"Query: {query}")
        
        result = interface.process_natural_language_query(query, db)
        print("Result:")
        print(f"  Response: {result.get('response', 'No response')}")
        print(f"  Function called: {result.get('function_called', 'None')}")
        print(f"  Data type: {type(result.get('data', 'None'))}")
        
        return True
        
    except Exception as e:
        print(f"OpenAI interface failed: {e}")
        print("\nTesting fallback interface...")
        
        # Test fallback interface
        fallback_interface = FallbackQueryInterface()
        result = fallback_interface.process_natural_language_query(query, db)
        print("Fallback Result:")
        print(f"  Response: {result.get('response', 'No response')}")
        print(f"  Function called: {result.get('function_called', 'None')}")
        
        return False
    
    finally:
        db.close()

if __name__ == "__main__":
    test_query_interface()
