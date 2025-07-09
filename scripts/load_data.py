#!/usr/bin/env python3
"""
Data loading script.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.graph_service import GraphService
from src.models import get_db

def main():
    csv_path = "scotiabank.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return 1
    
    print("Loading transaction data...")
    
    try:
        # Initialize services
        data_loader = DataLoader()
        graph_service = GraphService()
        
        # Get database session
        db = next(get_db())
        
        # Load data
        transactions = data_loader.process_csv_file(csv_path, db)
        print(f"Loaded {len(transactions)} transactions")
        
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
        for merchant, count in stats['top_merchants']:
            print(f"  {merchant}: {count} transactions")
        
        # Build graph
        print("\nBuilding transaction graph...")
        graph_service.build_graph_from_transactions(transactions, db)
        
        # Get graph statistics
        graph_stats = graph_service.get_graph_statistics()
        print("\nGraph Statistics:")
        for key, value in graph_stats.items():
            print(f"  {key}: {value}")
        
        # Train GNN
        print("\nTraining Graph Neural Network...")
        graph_service.train_gnn(transactions)
        
        # Detect clusters
        print("\nDetecting transaction clusters...")
        clusters = graph_service.detect_transaction_clusters(transactions, db, n_clusters=8)
        
        print(f"\nDetected {len(clusters)} clusters:")
        for cluster in clusters:
            print(f"  {cluster.name}: {len(cluster.transaction_ids)} transactions")
        
        db.close()
        print("\nData loading completed successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
