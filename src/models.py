from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
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
