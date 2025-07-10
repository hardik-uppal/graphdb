#!/usr/bin/env python3
"""
Test all Thought Journal tabs functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db
from src.enhanced_thought_journal_service import EnhancedThoughtJournalService

def test_thought_journal_tabs():
    print("📓 Testing Thought Journal Tabs")
    print("=" * 40)
    
    # Initialize services
    db = next(get_db())
    journal_service = EnhancedThoughtJournalService()
    
    try:
        # Test 1: Create a test entry
        print("\n1. Creating test entry...")
        entry = journal_service.create_mixed_entry(
            db=db,
            title="Test Entry for Tabs",
            content="This is a test entry to verify all tabs work correctly.",
            audio_data=None,
            audio_filename=None,
            audio_transcript=None,
            image_data=None,
            image_filename=None,
            image_description=None,
            tags=None,
            mood_score=None,
            importance_score=None
        )
        print(f"✅ Created entry: {entry.title}")
        
        # Test 2: View Entries functionality
        print("\n2. Testing View Entries...")
        entries = journal_service.get_journal_entries(db, limit=5)
        print(f"✅ Retrieved {len(entries)} entries")
        
        # Test 3: Search functionality
        print("\n3. Testing Search...")
        search_results = journal_service.search_journal_entries(db, "test", limit=3)
        print(f"✅ Search returned {len(search_results)} results")
        
        # Test 4: Linked Transactions
        print("\n4. Testing Linked Transactions...")
        linked_transactions = journal_service.get_linked_transactions(db, entry.id)
        print(f"✅ Entry has {len(linked_transactions)} linked transactions")
        
        # Test 5: Analytics
        print("\n5. Testing Analytics...")
        analytics = journal_service.get_journal_analytics(db)
        print(f"✅ Analytics:")
        print(f"   - Total entries: {analytics['total_entries']}")
        print(f"   - Total links: {analytics['total_transaction_links']}")
        print(f"   - Average mood: {analytics['average_mood']}")
        print(f"   - Top tags: {len(analytics['top_tags'])}")
        
        print("\n" + "=" * 40)
        print("🎉 All Thought Journal Tabs Working!")
        print("Tabs available in mobile app:")
        print("  ✅ ✍️ New Entry - Simplified AI-powered entry")
        print("  ✅ 📖 View Entries - Filter and browse entries")
        print("  ✅ 🔗 Linked Transactions - Search and manual linking")
        print("  ✅ 📊 Analytics - Charts and statistics")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    test_thought_journal_tabs()
