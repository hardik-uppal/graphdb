#!/usr/bin/env python3
"""
Data loading script with robust embedding repopulation.
"""

import sys
import os
import argparse
import time
from typing import List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.graph_service import GraphService
from src.models import get_db, Transaction
from src.embedding_service import EmbeddingService

def check_embedding_status(db):
    """Check the status of embeddings in the database."""
    print("Checking embedding status...")
    
    total_transactions = db.query(Transaction).count()
    
    # Count transactions with null embeddings
    null_embeddings = db.query(Transaction).filter(
        Transaction.embedding.is_(None)
    ).count()
    
    # Count transactions with zero embeddings (failed embeddings)
    # This is a simplified check - in practice, we'll check for embeddings that are all zeros
    zero_embeddings = 0  # We'll count these differently later
    
    # Alternative: count transactions with embeddings that are all zeros
    all_transactions = db.query(Transaction).filter(
        Transaction.embedding.isnot(None)
    ).all()
    
    for t in all_transactions:
        if t.embedding and all(x == 0.0 for x in t.embedding):
            zero_embeddings += 1
    
    # Count transactions with valid embeddings
    valid_embeddings = total_transactions - null_embeddings - zero_embeddings
    
    print(f"Embedding Status:")
    print(f"  Total transactions: {total_transactions}")
    print(f"  Valid embeddings: {valid_embeddings}")
    print(f"  Missing embeddings (null): {null_embeddings}")
    print(f"  Zero embeddings (failed): {zero_embeddings}")
    print(f"  Embeddings needed: {null_embeddings + zero_embeddings}")
    
    return {
        'total': total_transactions,
        'valid': valid_embeddings,
        'null': null_embeddings,
        'zero': zero_embeddings,
        'needed': null_embeddings + zero_embeddings
    }

def repopulate_embeddings(db, batch_size: int = 50, max_retries: int = 3, 
                         delay_between_batches: float = 1.0):
    """Repopulate embeddings for transactions with missing or zero embeddings."""
    print("Starting embedding repopulation...")
    
    embedding_service = EmbeddingService()
    
    # Find transactions that need embeddings
    transactions_needing_embeddings = db.query(Transaction).filter(
        Transaction.embedding.is_(None)
    ).all()
    
    # Also include transactions with zero embeddings
    all_embedded_transactions = db.query(Transaction).filter(
        Transaction.embedding.isnot(None)
    ).all()
    
    zero_embedding_transactions = []
    for t in all_embedded_transactions:
        if t.embedding and all(x == 0.0 for x in t.embedding):
            zero_embedding_transactions.append(t)
    
    # Combine both lists
    transactions_needing_embeddings.extend(zero_embedding_transactions)
    
    if not transactions_needing_embeddings:
        print("No transactions need embedding repopulation.")
        return True
    
    print(f"Found {len(transactions_needing_embeddings)} transactions needing embeddings")
    
    success_count = 0
    failed_count = 0
    
    # Process in batches
    for i in range(0, len(transactions_needing_embeddings), batch_size):
        batch = transactions_needing_embeddings[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(transactions_needing_embeddings) - 1) // batch_size + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} transactions)")
        
        # Prepare texts for this batch
        texts = [embedding_service.prepare_transaction_text(t) for t in batch]
        
        # Try to create embeddings with retries
        embeddings = None
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries} for batch {batch_num}")
                embeddings = embedding_service.create_embeddings_batch(texts)
                
                # Check if we got valid embeddings (not all zeros)
                valid_embeddings = []
                for emb in embeddings:
                    if emb and any(x != 0.0 for x in emb):
                        valid_embeddings.append(emb)
                    else:
                        valid_embeddings.append(None)
                
                if any(emb is not None for emb in valid_embeddings):
                    embeddings = valid_embeddings
                    break
                else:
                    print(f"  Got zero embeddings, retrying...")
                    embeddings = None
                    
            except Exception as e:
                print(f"  Error in attempt {attempt + 1}: {e}")
                embeddings = None
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * delay_between_batches
                    print(f"  Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
        
        # Update database with results
        if embeddings:
            batch_success = 0
            batch_failed = 0
            
            for transaction, embedding in zip(batch, embeddings):
                if embedding is not None:
                    transaction.embedding = embedding
                    batch_success += 1
                else:
                    batch_failed += 1
                    print(f"  Failed to generate embedding for transaction {transaction.id}")
            
            try:
                db.commit()
                success_count += batch_success
                failed_count += batch_failed
                print(f"  Batch {batch_num} completed: {batch_success} success, {batch_failed} failed")
            except Exception as e:
                print(f"  Error committing batch {batch_num}: {e}")
                db.rollback()
                failed_count += len(batch)
        else:
            print(f"  Batch {batch_num} completely failed")
            failed_count += len(batch)
        
        # Small delay between batches to avoid rate limiting
        if i + batch_size < len(transactions_needing_embeddings):
            time.sleep(delay_between_batches)
    
    print(f"\nEmbedding repopulation completed:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total processed: {success_count + failed_count}")
    
    return failed_count == 0

def main():
    parser = argparse.ArgumentParser(description="Load data and manage embeddings")
    parser.add_argument("--repopulate-only", action="store_true",
                       help="Only repopulate embeddings, don't load new data")
    parser.add_argument("--check-status", action="store_true",
                       help="Check embedding status and exit")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for embedding processing")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for failed batches")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between batches in seconds")
    
    args = parser.parse_args()
    
    csv_path = "scotiabank.csv"
    
    try:
        # Get database session
        db = next(get_db())
        
        if args.check_status:
            check_embedding_status(db)
            db.close()
            return 0
        
        if args.repopulate_only:
            # Only repopulate embeddings
            status = check_embedding_status(db)
            if status['needed'] > 0:
                success = repopulate_embeddings(db, args.batch_size, args.max_retries, args.delay)
                if success:
                    print("\nEmbedding repopulation completed successfully!")
                else:
                    print("\nEmbedding repopulation completed with some failures.")
                    print("You may want to run again to retry failed embeddings.")
                
                # Check status again
                print("\nFinal embedding status:")
                check_embedding_status(db)
            else:
                print("No embeddings need repopulation.")
            
            db.close()
            return 0
        
        # Full data loading pipeline
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return 1
        
        print("Loading transaction data...")
        
        # Initialize services
        data_loader = DataLoader()
        graph_service = GraphService()
        
        # Load data
        transactions = data_loader.process_csv_file(csv_path, db)
        print(f"Loaded {len(transactions)} transactions")
        
        # Check and repopulate embeddings if needed
        print("\nChecking embedding status after initial load...")
        status = check_embedding_status(db)
        
        if status['needed'] > 0:
            print(f"Repopulating {status['needed']} missing embeddings...")
            repopulate_embeddings(db, args.batch_size, args.max_retries, args.delay)
            
            # Refresh transactions to get new embeddings
            for transaction in transactions:
                db.refresh(transaction)
        
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
