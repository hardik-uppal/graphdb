#!/usr/bin/env python3
"""
Demo script showing the Graph-Powered Transaction Analytics capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db, Transaction
from src.query_interface import QueryInterface
from src.graph_service import GraphService
from src.embedding_service import EmbeddingService
import json

def demo_natural_language_queries():
    """Demonstrate natural language query capabilities."""
    print("🗣️  Natural Language Query Demo")
    print("=" * 50)
    
    db = next(get_db())
    interface = QueryInterface()
    
    # Sample queries
    queries = [
        "What did I spend the most money on?",
        "Show me my top 5 merchants by spending",
        "What's my total spending this month?",
        "Find transactions similar to investment",
        "Detect any anomalous transactions"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        try:
            result = interface.process_natural_language_query(query, db)
            print(f"Response: {result.get('response', 'No response')}")
            
            if result.get('data'):
                print(f"Function called: {result.get('function_called', 'Unknown')}")
                # Show a preview of the data
                data = result['data']
                if isinstance(data, dict) and 'total_transactions' in data:
                    print(f"Found {data['total_transactions']} transactions")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"Returned {len(data)} results")
        
        except Exception as e:
            print(f"Error: {e}")
    
    db.close()

def demo_graph_analysis():
    """Demonstrate graph analysis capabilities."""
    print("\n\n🕸️  Graph Analysis Demo")
    print("=" * 50)
    
    db = next(get_db())
    graph_service = GraphService()
    
    # Load transactions
    transactions = db.query(Transaction).limit(100).all()  # Limit for demo
    
    if not transactions:
        print("No transactions found. Please run load_data.py first.")
        return
    
    print(f"Analyzing {len(transactions)} transactions...")
    
    # Build graph
    print("\n1. Building transaction graph...")
    graph_service.build_graph_from_transactions(transactions, db)
    
    # Get statistics
    stats = graph_service.get_graph_statistics()
    print(f"   - Nodes: {stats.get('nodes', 0)}")
    print(f"   - Edges: {stats.get('edges', 0)}")
    print(f"   - Density: {stats.get('density', 0):.3f}")
    print(f"   - Connected components: {stats.get('connected_components', 0)}")
    
    # Detect clusters
    print("\n2. Detecting transaction clusters...")
    clusters = graph_service.detect_transaction_clusters(transactions, db, n_clusters=5)
    
    for cluster in clusters:
        print(f"   - {cluster.name}: {len(cluster.transaction_ids)} transactions")
    
    # Find anomalies
    print("\n3. Finding anomalous transactions...")
    anomalies = graph_service.find_anomalous_transactions(transactions, threshold_percentile=90)
    
    print(f"   Found {len(anomalies)} potentially anomalous transactions")
    for anomaly in anomalies[:3]:  # Show first 3
        print(f"   - ${anomaly.amount:.2f} on {anomaly.date.strftime('%Y-%m-%d')} ({anomaly.transaction_type})")
    
    db.close()

def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("\n\n🔍 Semantic Search Demo")
    print("=" * 50)
    
    db = next(get_db())
    embedding_service = EmbeddingService()
    
    # Sample searches
    searches = [
        "coffee and food purchases",
        "investment and savings",
        "utility bills and payments",
        "transportation and gas"
    ]
    
    for i, search_term in enumerate(searches, 1):
        print(f"\n{i}. Searching for: '{search_term}'")
        print("-" * 40)
        
        try:
            results = embedding_service.semantic_search(search_term, db, limit=3)
            
            if results:
                print(f"Found {len(results)} similar transactions:")
                for result in results:
                    print(f"   - ${result.amount:.2f} | {result.transaction_type} | {result.merchant}")
            else:
                print("   No similar transactions found")
        
        except Exception as e:
            print(f"   Error: {e}")
    
    db.close()

def demo_data_insights():
    """Show basic data insights."""
    print("\n\n📊 Data Insights Demo")
    print("=" * 50)
    
    db = next(get_db())
    
    # Basic statistics
    transactions = db.query(Transaction).all()
    
    if not transactions:
        print("No transactions found. Please run load_data.py first.")
        return
    
    total_income = sum(t.amount for t in transactions if t.amount > 0)
    total_expenses = sum(t.amount for t in transactions if t.amount < 0)
    
    print(f"Total transactions: {len(transactions)}")
    print(f"Total income: ${total_income:,.2f}")
    print(f"Total expenses: ${abs(total_expenses):,.2f}")
    print(f"Net amount: ${total_income + total_expenses:,.2f}")
    
    # Top categories
    categories = {}
    for t in transactions:
        if t.transaction_type and t.amount < 0:
            categories[t.transaction_type] = categories.get(t.transaction_type, 0) + abs(t.amount)
    
    print(f"\nTop spending categories:")
    top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
    for category, amount in top_categories:
        print(f"   - {category}: ${amount:,.2f}")
    
    # Top merchants
    merchants = {}
    for t in transactions:
        if t.merchant and t.amount < 0:
            merchants[t.merchant] = merchants.get(t.merchant, 0) + abs(t.amount)
    
    print(f"\nTop merchants:")
    top_merchants = sorted(merchants.items(), key=lambda x: x[1], reverse=True)[:5]
    for merchant, amount in top_merchants:
        print(f"   - {merchant.strip()}: ${amount:,.2f}")
    
    db.close()

def main():
    print("🚀 Graph-Powered Transaction Analytics Demo")
    print("=" * 60)
    print("This demo showcases the key capabilities of the system.")
    print("Make sure you've run 'python scripts/load_data.py' first!")
    
    try:
        # Run demos
        demo_data_insights()
        demo_semantic_search()
        demo_graph_analysis()
        demo_natural_language_queries()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\nTo explore more features, run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure:")
        print("1. PostgreSQL is running")
        print("2. Database is initialized (python scripts/init_db.py)")
        print("3. Data is loaded (python scripts/load_data.py)")
        print("4. OpenAI API key is configured in .env")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
