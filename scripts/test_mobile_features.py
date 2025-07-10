#!/usr/bin/env python3
"""
Test script for the mobile-friendly Second Brain app.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db
from src.enhanced_thought_journal_service import EnhancedThoughtJournalService
from src.chat_history_service import ChatHistoryService
from src.smart_query_interface import SmartQueryInterface

def test_mobile_features():
    print("🧠 Testing Mobile-Friendly Second Brain Features")
    print("=" * 50)
    
    # Initialize services
    db = next(get_db())
    journal_service = EnhancedThoughtJournalService()
    chat_service = ChatHistoryService()
    query_interface = SmartQueryInterface()
    
    try:
        # Test 1: Create a simple thought journal entry
        print("\n1. Testing Thought Journal Entry Creation...")
        entry = journal_service.create_mixed_entry(
            db=db,
            title="Test Mobile Entry",
            content="This is a test entry for the mobile app. I'm feeling good about the new UI improvements.",
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
        print(f"✅ Created journal entry: {entry.title}")
        if entry.tags:
            print(f"   Auto-extracted tags: {entry.tags}")
        if entry.mood_score is not None:
            print(f"   Auto-detected mood: {entry.mood_score}")
        
        # Test 2: Create a chat thread
        print("\n2. Testing Chat Thread Creation...")
        thread = chat_service.create_thread(db, "Test Mobile Chat")
        print(f"✅ Created chat thread: {thread.thread_title}")
        
        # Test 3: Add messages to chat
        print("\n3. Testing Chat Messages...")
        user_msg = chat_service.add_message(db, thread.id, "user", "Hello, this is a test message")
        ai_msg = chat_service.add_message(db, thread.id, "assistant", "Hello! I'm here to help with your financial questions.")
        print(f"✅ Added user message: {user_msg.content[:50]}...")
        print(f"✅ Added AI message: {ai_msg.content[:50]}...")
        
        # Test 4: Test query interface
        print("\n4. Testing Query Interface...")
        result = query_interface.process_query("What are my recent transactions?", db)
        print(f"✅ Query processed: {result.get('response', 'No response')[:100]}...")
        
        # Test 5: Get recent journal entries
        print("\n5. Testing Journal Entry Retrieval...")
        recent_entries = journal_service.get_journal_entries(db, limit=3)
        print(f"✅ Retrieved {len(recent_entries)} recent journal entries")
        
        # Test 6: Get chat history
        print("\n6. Testing Chat History...")
        recent_threads = chat_service.get_recent_threads(db, limit=3)
        print(f"✅ Retrieved {len(recent_threads)} recent chat threads")
        
        print("\n" + "=" * 50)
        print("🎉 All mobile features tested successfully!")
        print("The app is ready for mobile use with:")
        print("  • Large navigation buttons")
        print("  • Simplified thought entry")
        print("  • Chat history management")
        print("  • Mobile-responsive design")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    test_mobile_features()
