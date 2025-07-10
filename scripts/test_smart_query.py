#!/usr/bin/env python3
"""
Test script for the new SmartQueryInterface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_query_interface import SmartQueryInterface
from src.models import get_db
import json

def test_smart_queries():
    """Test various types of queries with the new smart interface."""
    
    # Initialize
    db = next(get_db())
    smart_query = SmartQueryInterface()
    
    # Test queries
    test_queries = [
        "How much did I spend on groceries last month?",
        "Find all transactions at Costco",
        "What are my biggest expenses?",
        "Show me all Honda payments",
        "How much did I spend on gas stations?",
        "Find transactions over $1000",
        "What's my total income this month?",
        "Show me all Wealthsimple investments",
        "Find unusual or large transactions",
        "What did I spend on restaurants?"
    ]
    
    print("=== Testing Smart Query Interface ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = smart_query.process_natural_language_query(query, db)
            
            print(f"Method: {result.get('method', 'unknown')}")
            print(f"Transactions found: {result.get('transactions_found', 0)}")
            print(f"Response: {result.get('response', 'No response')}")
            
            # Show some analysis results if available
            if 'analysis_results' in result:
                analysis = result['analysis_results']
                if 'total_amount' in analysis:
                    print(f"Total amount: ${analysis['total_amount']:.2f}")
                if 'top_merchants' in analysis and analysis['top_merchants']:
                    print(f"Top merchant: {analysis['top_merchants'][0][0]}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*70 + "\n")
    
    db.close()

if __name__ == "__main__":
    test_smart_queries()
