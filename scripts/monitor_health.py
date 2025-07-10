#!/usr/bin/env python3
"""
Monitoring script to check embedding and system health.
"""

import sys
import os
import time
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db, Transaction
from src.embedding_service import EmbeddingService
from src.config import Config

def get_system_health():
    """Get overall system health information."""
    health = {
        'timestamp': datetime.now().isoformat(),
        'database': 'unknown',
        'openai_api': 'unknown',
        'embeddings': {
            'total': 0,
            'valid': 0,
            'null': 0,
            'zero': 0
        }
    }
    
    try:
        # Check database connectivity
        db = next(get_db())
        health['database'] = 'connected'
        
        # Check transaction count
        total_transactions = db.query(Transaction).count()
        health['embeddings']['total'] = total_transactions
        
        # Check null embeddings
        null_embeddings = db.query(Transaction).filter(
            Transaction.embedding.is_(None)
        ).count()
        health['embeddings']['null'] = null_embeddings
        
        # Check zero embeddings
        zero_embeddings = 0
        embedded_transactions = db.query(Transaction).filter(
            Transaction.embedding.isnot(None)
        ).all()
        
        for t in embedded_transactions:
            if t.embedding and all(x == 0.0 for x in t.embedding):
                zero_embeddings += 1
        
        health['embeddings']['zero'] = zero_embeddings
        health['embeddings']['valid'] = total_transactions - null_embeddings - zero_embeddings
        
        db.close()
        
    except Exception as e:
        health['database'] = f'error: {str(e)}'
    
    try:
        # Test OpenAI API
        embedding_service = EmbeddingService()
        test_embedding = embedding_service.create_embedding("test")
        
        if test_embedding and any(x != 0.0 for x in test_embedding):
            health['openai_api'] = 'working'
        else:
            health['openai_api'] = 'returning_zeros'
            
    except Exception as e:
        health['openai_api'] = f'error: {str(e)}'
    
    return health

def print_health_report(health, detailed=False):
    """Print a formatted health report."""
    print(f"=== System Health Report - {health['timestamp']} ===")
    print(f"Database: {health['database']}")
    print(f"OpenAI API: {health['openai_api']}")
    
    emb = health['embeddings']
    print(f"\nEmbeddings:")
    print(f"  Total transactions: {emb['total']}")
    print(f"  Valid embeddings: {emb['valid']}")
    print(f"  Null embeddings: {emb['null']}")
    print(f"  Zero embeddings: {emb['zero']}")
    
    if emb['total'] > 0:
        valid_pct = (emb['valid'] / emb['total']) * 100
        print(f"  Coverage: {valid_pct:.1f}%")
    
    # Health indicators
    issues = []
    if health['database'] != 'connected':
        issues.append("Database connectivity issue")
    if health['openai_api'] not in ['working']:
        issues.append("OpenAI API issue")
    if emb['null'] > 0:
        issues.append(f"{emb['null']} transactions missing embeddings")
    if emb['zero'] > 0:
        issues.append(f"{emb['zero']} transactions with failed embeddings")
    
    if issues:
        print(f"\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ All systems healthy")
    
    if detailed:
        print(f"\nConfiguration:")
        print(f"  Embedding dimension: {Config.EMBEDDING_DIMENSION}")
        print(f"  Similarity threshold: {Config.SIMILARITY_THRESHOLD}")
        print(f"  Database URL: {Config.DATABASE_URL[:50]}...")

def save_health_log(health, log_file="health_log.json"):
    """Save health data to a log file."""
    try:
        # Try to read existing log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'entries': []}
        
        # Add new entry
        log_data['entries'].append(health)
        
        # Keep only last 100 entries
        if len(log_data['entries']) > 100:
            log_data['entries'] = log_data['entries'][-100:]
        
        # Save back
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not save health log: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system health")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--continuous", action="store_true",
                       help="Run continuously")
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval in seconds (for continuous mode)")
    parser.add_argument("--log", action="store_true",
                       help="Save health data to log file")
    parser.add_argument("--json", action="store_true",
                       help="Output in JSON format")
    
    args = parser.parse_args()
    
    try:
        if args.continuous:
            print(f"Starting continuous monitoring (interval: {args.interval}s)")
            print("Press Ctrl+C to stop")
            
            while True:
                health = get_system_health()
                
                if args.json:
                    print(json.dumps(health, indent=2))
                else:
                    print_health_report(health, args.detailed)
                
                if args.log:
                    save_health_log(health)
                
                if not args.json:
                    print(f"\nNext check in {args.interval} seconds...\n")
                
                time.sleep(args.interval)
        else:
            # Single check
            health = get_system_health()
            
            if args.json:
                print(json.dumps(health, indent=2))
            else:
                print_health_report(health, args.detailed)
            
            if args.log:
                save_health_log(health)
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
