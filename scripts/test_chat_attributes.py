#!/usr/bin/env python3
"""
Quick test to verify chat functionality works with correct attributes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db
from src.chat_history_service import ChatHistoryService

def test_chat_attributes():
    print("🧪 Testing Chat Attributes Fix")
    print("=" * 40)
    
    # Initialize services
    db = next(get_db())
    chat_service = ChatHistoryService()
    
    try:
        # Create a test thread
        thread = chat_service.create_thread(db, "Test Thread")
        print(f"✅ Created thread: {thread.thread_title}")
        
        # Add messages
        user_msg = chat_service.add_message(db, thread.id, "user", "Test user message")
        ai_msg = chat_service.add_message(db, thread.id, "assistant", "Test AI response")
        
        print(f"✅ User message - type: {user_msg.message_type}, content: {user_msg.content}")
        print(f"✅ AI message - type: {ai_msg.message_type}, content: {ai_msg.content}")
        
        # Get messages back
        messages = chat_service.get_messages(db, thread.id)
        print(f"✅ Retrieved {len(messages)} messages")
        
        for msg in messages:
            print(f"   - {msg.message_type}: {msg.content}")
        
        # Get threads
        threads = chat_service.get_recent_threads(db, limit=3)
        print(f"✅ Retrieved {len(threads)} threads")
        
        for t in threads:
            print(f"   - {t.thread_title}")
        
        print("\n" + "=" * 40)
        print("🎉 All chat attributes working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    test_chat_attributes()
