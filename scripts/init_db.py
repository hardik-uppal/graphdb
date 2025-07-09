#!/usr/bin/env python3
"""
Database initialization script.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_tables, get_db
from src.config import Config

def main():
    print("Initializing database...")
    print(f"Database URL: {Config.DATABASE_URL}")
    
    try:
        # Create tables
        create_tables()
        print("Database tables created successfully!")
        
        # Test connection
        db = next(get_db())
        print("Database connection test successful!")
        db.close()
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        print("Make sure PostgreSQL is running and the database exists.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
