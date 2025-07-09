#!/usr/bin/env python3
"""
Demo data loading script that works without OpenAI embeddings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.models import get_db, Transaction
import numpy as np

def create_mock_embeddings(transactions):
    """Create simple mock embeddings based on transaction features."""
    print("Creating mock embeddings (OpenAI API not available)...")
    
    for transaction in transactions:
        # Create a simple feature vector based on transaction properties
        features = []
        
        # Amount-based features
        features.append(float(transaction.amount))
        features.append(1.0 if transaction.amount > 0 else -1.0)  # income/expense flag
        features.append(abs(transaction.amount) / 1000.0)  # normalized amount
        
        # Type-based features (simple hash)
        type_hash = hash(transaction.transaction_type or "") % 100 / 100.0
        features.append(type_hash)
        
        # Merchant-based features
        merchant_hash = hash(transaction.merchant or "") % 100 / 100.0
        features.append(merchant_hash)
        
        # Date-based features
        features.append(float(transaction.date.day) / 31.0)
        features.append(float(transaction.date.month) / 12.0)
        
        # Pad with zeros to match expected embedding dimension (1536)
        while len(features) < 1536:
            features.append(0.0)
        
        # Truncate if needed
        features = features[:1536]
        
        transaction.embedding = features

def main():
    csv_path = "scotiabank.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return 1
    
    print("Loading transaction data without OpenAI embeddings...")
    
    try:
        # Initialize services
        data_loader = DataLoader()
        
        # Get database session
        db = next(get_db())
        
        # Check if data already exists
        existing_transactions = db.query(Transaction).all()
        
        if existing_transactions:
            print(f"Found {len(existing_transactions)} existing transactions")
            transactions = existing_transactions
        else:
            # Load data
            df = data_loader.parse_scotiabank_csv(csv_path)
            transactions = data_loader.load_to_database(df, db)
        
        # Create mock embeddings for all transactions without them
        transactions_without_embeddings = [t for t in transactions if not t.embedding]
        if transactions_without_embeddings:
            create_mock_embeddings(transactions_without_embeddings)
            db.commit()
            print(f"Created mock embeddings for {len(transactions_without_embeddings)} transactions")
        
        # Get summary statistics
        stats = data_loader.get_summary_statistics(transactions)
        print("\nSummary Statistics:")
        print(f"Total transactions: {stats['total_transactions']}")
        print(f"Total income: ${stats['total_income']:,.2f}")
        print(f"Total expenses: ${stats['total_expenses']:,.2f}")
        print(f"Net amount: ${stats['net_amount']:,.2f}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        
        print("\nTop transaction types:")
        for type_name, count in stats['top_transaction_types']:
            print(f"  {type_name}: {count} transactions")
        
        print("\nTop merchants:")
        for merchant, count in stats['top_merchants'][:5]:  # Limit to top 5
            print(f"  {merchant}: {count} transactions")
        
        db.close()
        print("\nDemo data loading completed successfully!")
        print("\nYou can now run: streamlit run app.py")
        print("Note: Some features requiring OpenAI embeddings will use mock data.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
