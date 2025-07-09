import openai
import numpy as np
from typing import List, Dict, Any
import asyncio
from sqlalchemy.orm import Session

from .config import Config
from .models import Transaction, get_db

class EmbeddingService:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * Config.EMBEDDING_DIMENSION
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error creating batch embeddings: {e}")
            return [[0.0] * Config.EMBEDDING_DIMENSION] * len(texts)
    
    def prepare_transaction_text(self, transaction: Transaction) -> str:
        """Prepare transaction text for embedding."""
        parts = []
        
        if transaction.transaction_type:
            parts.append(f"Type: {transaction.transaction_type.strip()}")
        
        if transaction.merchant:
            parts.append(f"Merchant: {transaction.merchant.strip()}")
        
        # Add amount context
        amount_desc = "income" if transaction.amount > 0 else "expense"
        parts.append(f"Amount: {amount_desc} ${abs(transaction.amount):.2f}")
        
        return " | ".join(parts)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
            
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def find_similar_transactions(self, target_embedding: List[float], 
                                 transactions: List[Transaction], 
                                 threshold: float = None) -> List[tuple]:
        """Find transactions similar to target embedding."""
        if threshold is None:
            threshold = Config.SIMILARITY_THRESHOLD
            
        similarities = []
        for transaction in transactions:
            if transaction.embedding:
                similarity = self.cosine_similarity(target_embedding, transaction.embedding)
                if similarity >= threshold:
                    similarities.append((transaction, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def update_transaction_embeddings(self, db: Session):
        """Update embeddings for transactions that don't have them."""
        transactions_without_embeddings = db.query(Transaction).filter(
            Transaction.embedding.is_(None)
        ).all()
        
        if not transactions_without_embeddings:
            return
        
        print(f"Creating embeddings for {len(transactions_without_embeddings)} transactions...")
        
        # Prepare texts
        texts = [self.prepare_transaction_text(t) for t in transactions_without_embeddings]
        
        # Create embeddings in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_transactions = transactions_without_embeddings[i:i + batch_size]
            
            embeddings = self.create_embeddings_batch(batch_texts)
            
            # Update transactions
            for transaction, embedding in zip(batch_transactions, embeddings):
                transaction.embedding = embedding
            
            db.commit()
            print(f"Updated embeddings for batch {i//batch_size + 1}")
        
        print("All embeddings updated!")
    
    def semantic_search(self, query: str, db: Session, limit: int = 10) -> List[Transaction]:
        """Search transactions using semantic similarity."""
        query_embedding = self.create_embedding(query)
        
        all_transactions = db.query(Transaction).filter(
            Transaction.embedding.isnot(None)
        ).all()
        
        similarities = self.find_similar_transactions(
            query_embedding, all_transactions, threshold=0.3
        )
        
        return [t[0] for t in similarities[:limit]]
