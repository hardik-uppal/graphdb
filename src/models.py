from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
import json

from .config import Config

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(100), nullable=False)
    merchant = Column(String(200))
    description = Column(Text)
    embedding = Column(ARRAY(Float))  # Store embedding as array
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TransactionCluster(Base):
    __tablename__ = "transaction_clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    transaction_ids = Column(ARRAY(Integer))  # Array of transaction IDs
    centroid_embedding = Column(ARRAY(Float))
    
    created_at = Column(DateTime, default=datetime.utcnow)

class GraphEdge(Base):
    __tablename__ = "graph_edges"
    
    id = Column(Integer, primary_key=True, index=True)
    source_transaction_id = Column(Integer, nullable=False)
    target_transaction_id = Column(Integer, nullable=False)
    edge_type = Column(String(50), nullable=False)  # similarity, temporal, merchant
    weight = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ThoughtJournal(Base):
    __tablename__ = "thought_journals"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)  # Text content
    entry_type = Column(String(20), nullable=False)  # 'text', 'audio', 'image', 'mixed'
    
    # Audio data
    audio_data = Column(LargeBinary)  # Store audio file as binary
    audio_transcript = Column(Text)  # Transcribed text from audio
    audio_filename = Column(String(255))
    audio_duration = Column(Float)  # Duration in seconds
    
    # Image data
    image_data = Column(LargeBinary)  # Store image file as binary
    image_description = Column(Text)  # AI-generated description of image
    image_filename = Column(String(255))
    image_size = Column(String(50))  # e.g., "1920x1080"
    
    # Semantic similarity
    embedding = Column(ARRAY(Float))  # Combined embedding of all content
    
    # Metadata
    mood_score = Column(Float)  # -1 to 1 (negative to positive)
    importance_score = Column(Float)  # 0 to 1
    tags = Column(ARRAY(String))  # User-defined tags
    
    # Linking to transactions
    linked_transaction_ids = Column(ARRAY(Integer))  # Manually linked transactions
    auto_linked_transaction_ids = Column(ARRAY(Integer))  # Auto-linked based on similarity
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class JournalTransactionLink(Base):
    __tablename__ = "journal_transaction_links"
    
    id = Column(Integer, primary_key=True, index=True)
    journal_id = Column(Integer, nullable=False)
    transaction_id = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=False)  # 0 to 1
    link_type = Column(String(20), nullable=False)  # 'manual', 'auto', 'semantic'
    link_reason = Column(Text)  # Explanation of why they're linked
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatThread(Base):
    __tablename__ = "chat_threads"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_title = Column(String(200))  # Auto-generated or user-defined
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, nullable=False)
    message_type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float))  # For semantic search of chat history
    
    # Response metadata
    response_time = Column(Float)  # Time taken to generate response
    token_usage = Column(Integer)  # Tokens used
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection
engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)
