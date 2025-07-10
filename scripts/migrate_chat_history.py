#!/usr/bin/env python3
"""
Database migration script to add chat history tables.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from src.models import Base, ChatThread, ChatMessage
from src.config import Config

def create_chat_tables():
    """Create the chat history tables."""
    
    print("Creating chat history tables...")
    
    engine = create_engine(Config.DATABASE_URL)
    
    try:
        # Create tables
        Base.metadata.create_all(bind=engine, tables=[
            ChatThread.__table__,
            ChatMessage.__table__
        ])
        
        print("✓ Chat history tables created successfully!")
        
        # Verify tables exist
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('chat_threads', 'chat_messages')
            """))
            
            tables = [row[0] for row in result]
            print(f"✓ Created tables: {', '.join(tables)}")
    
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = create_chat_tables()
    if success:
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)
