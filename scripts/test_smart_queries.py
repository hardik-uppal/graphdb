#!/usr/bin/env python3
"""
Test script to demonstrate the new Smart Query Interface capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_query_interface import SmartQueryInterface
from src.models import get_db

def test_smart_queries():
    """Test various types of queries to show the improvement."""
    
    print("=== Testing Smart Query Interface ===\n")
    
    # Initialize
    db = next(get_db())
    qi = SmartQueryInterface()
    
    # Test queries that show the improvement
    test_queries = [
        "How much did I spend on groceries last month?",
        "Show me all transactions over $100",
        "What are my spending patterns by day of the week?",
        "Find any unusual or anomalous transactions",
        "Compare my spending on restaurants vs groceries",
        "What's my average transaction amount?",
        "Show me my biggest expenses this year",
        "Find transactions similar to 'coffee shop purchase'",
        "What's the trend in my monthly spending?",
        "Who are my top 5 merchants by spending?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        try:
            result = qi.process_query(query, db)
            
            if result['success']:
                print("✓ Query processed successfully")
                print(f"Response: {result['response'][:300]}...")
                
                # Show some context info
                context = result['context']
                print(f"Relevant transactions: {len(context['relevant_transactions'])}")
                print(f"Query type: {context['query_type']}")
                
                if context['anomalies']:
                    print(f"Anomalies detected: {len(context['anomalies'])}")
                    
            else:
                print("✗ Query failed")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
        
        print("-" * 60)
    
    db.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_smart_queries()
