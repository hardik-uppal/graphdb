#!/usr/bin/env python3
"""
Embedding management script for checking and repopulating embeddings.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db, Transaction
from src.embedding_service import EmbeddingService
from src.graph_service import GraphService

def check_embedding_status(db):
    """Check the status of embeddings in the database."""
    print("Checking embedding status...")
    
    total_transactions = db.query(Transaction).count()
    
    # Count transactions with null embeddings
    null_embeddings = db.query(Transaction).filter(
        Transaction.embedding.is_(None)
    ).count()
    
    # Count transactions with zero embeddings
    zero_embeddings = 0
    embedded_transactions = db.query(Transaction).filter(
        Transaction.embedding.isnot(None)
    ).all()
    
    for t in embedded_transactions:
        if t.embedding and all(x == 0.0 for x in t.embedding):
            zero_embeddings += 1
    
    valid_embeddings = total_transactions - null_embeddings - zero_embeddings
    
    print(f"\nEmbedding Status:")
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

def test_single_embedding():
    """Test creating a single embedding to verify API connectivity."""
    print("Testing single embedding creation...")
    
    embedding_service = EmbeddingService()
    
    test_text = "Type: Purchase | Merchant: Test Store | Amount: expense $25.00"
    
    try:
        embedding = embedding_service.create_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            if any(x != 0.0 for x in embedding):
                print("✓ Single embedding test successful")
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  First 5 values: {embedding[:5]}")
                return True
            else:
                print("✗ Single embedding test failed: got zero embedding")
                return False
        else:
            print("✗ Single embedding test failed: no embedding returned")
            return False
            
    except Exception as e:
        print(f"✗ Single embedding test failed: {e}")
        return False

def repopulate_embeddings(db, batch_size=50, max_retries=3, delay=1.0):
    """Repopulate embeddings using the EmbeddingService."""
    embedding_service = EmbeddingService()
    
    result = embedding_service.repopulate_embeddings(
        db, batch_size, max_retries, delay
    )
    
    print(f"\nRepopulation Results:")
    print(f"  Total processed: {result['total_processed']}")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Already had embeddings: {result['already_had_embeddings']}")
    
    return result['failed'] == 0

def rebuild_graph(db):
    """Rebuild the graph after embedding repopulation."""
    print("Rebuilding graph with updated embeddings...")
    
    graph_service = GraphService()
    
    # Get all transactions
    transactions = db.query(Transaction).all()
    
    # Rebuild graph
    graph_service.build_graph_from_transactions(transactions, db)
    
    # Get statistics
    stats = graph_service.get_graph_statistics()
    
    print("Graph rebuilt successfully!")
    print("Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Manage transaction embeddings")
    parser.add_argument("--check", action="store_true",
                       help="Check embedding status")
    parser.add_argument("--test", action="store_true",
                       help="Test single embedding creation")
    parser.add_argument("--repopulate", action="store_true",
                       help="Repopulate missing embeddings")
    parser.add_argument("--rebuild-graph", action="store_true",
                       help="Rebuild graph after repopulation")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for processing")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for failed batches")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between batches")
    
    args = parser.parse_args()
    
    # Default action is to check status
    if not any([args.check, args.test, args.repopulate, args.rebuild_graph]):
        args.check = True
    
    try:
        db = next(get_db())
        
        if args.test:
            success = test_single_embedding()
            if not success:
                print("Single embedding test failed. Check your OpenAI API key and quota.")
                return 1
        
        if args.check:
            status = check_embedding_status(db)
            
            if status['needed'] > 0:
                print(f"\nRecommendation: Run with --repopulate to fix {status['needed']} missing embeddings")
            else:
                print("\n✓ All embeddings are present")
        
        if args.repopulate:
            status = check_embedding_status(db)
            
            if status['needed'] > 0:
                print(f"\nStarting repopulation of {status['needed']} embeddings...")
                success = repopulate_embeddings(db, args.batch_size, args.max_retries, args.delay)
                
                if success:
                    print("\n✓ Repopulation completed successfully!")
                    
                    # Check status again
                    print("\nFinal status:")
                    check_embedding_status(db)
                    
                    if args.rebuild_graph:
                        rebuild_graph(db)
                else:
                    print("\n⚠ Repopulation completed with some failures")
                    print("You may want to run again to retry failed embeddings")
            else:
                print("No embeddings need repopulation")
        
        elif args.rebuild_graph:
            rebuild_graph(db)
        
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
