#!/usr/bin/env python3
"""
Test script for the Thought Journal functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.thought_journal_service import ThoughtJournalService
from src.models import get_db
from datetime import datetime

def test_thought_journal():
    """Test the thought journal functionality."""
    
    print("=== Testing Thought Journal Service ===\n")
    
    # Initialize
    db = next(get_db())
    service = ThoughtJournalService()
    
    try:
        # Test 1: Create a text entry
        print("Test 1: Creating text entry...")
        text_entry = service.create_text_entry(
            db=db,
            title="My First Journal Entry",
            content="Today I spent money on groceries and coffee. I'm trying to be more mindful of my spending habits.",
            tags=["spending", "reflection", "groceries"],
            mood_score=0.3,
            importance_score=0.7
        )
        print(f"✓ Created text entry: {text_entry.id}")
        print(f"  Auto-linked to {len(text_entry.auto_linked_transaction_ids or [])} transactions")
        
        # Test 2: Create an image entry
        print("\nTest 2: Creating image entry...")
        # Create a simple test image (1x1 pixel)
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
        
        image_entry = service.create_image_entry(
            db=db,
            title="Receipt Photo",
            image_data=test_image_data,
            image_filename="receipt.png",
            description="A receipt from my grocery shopping trip today",
            tags=["receipt", "groceries", "documentation"],
            mood_score=0.0,
            importance_score=0.8
        )
        print(f"✓ Created image entry: {image_entry.id}")
        print(f"  Auto-linked to {len(image_entry.auto_linked_transaction_ids or [])} transactions")
        
        # Test 3: Search entries
        print("\nTest 3: Searching entries...")
        search_results = service.search_journal_entries(db, "groceries", limit=5)
        print(f"✓ Found {len(search_results)} entries matching 'groceries'")
        
        for entry, similarity in search_results:
            print(f"  - {entry.title} (similarity: {similarity:.3f})")
        
        # Test 4: Get analytics
        print("\nTest 4: Getting analytics...")
        analytics = service.get_journal_analytics(db)
        print(f"✓ Total entries: {analytics['total_entries']}")
        print(f"  Entries by type: {analytics['entries_by_type']}")
        print(f"  Top tags: {analytics['top_tags'][:3]}")
        
        # Test 5: Manual linking
        print("\nTest 5: Testing manual linking...")
        from src.models import Transaction
        
        # Get a transaction
        transaction = db.query(Transaction).first()
        if transaction:
            link = service.manually_link_to_transaction(
                db, text_entry.id, transaction.id, "Testing manual link"
            )
            print(f"✓ Manually linked entry {text_entry.id} to transaction {transaction.id}")
            print(f"  Link similarity: {link.similarity_score:.3f}")
        
        # Test 6: Get linked transactions
        print("\nTest 6: Getting linked transactions...")
        linked_transactions = service.get_linked_transactions(db, text_entry.id)
        print(f"✓ Found {len(linked_transactions)} linked transactions")
        
        for link_data in linked_transactions:
            transaction = link_data['transaction']
            link = link_data['link']
            print(f"  - {transaction.date}: ${transaction.amount:.2f} ({link.link_type})")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_thought_journal()
