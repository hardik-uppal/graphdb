import openai
import numpy as np
from typing import List, Dict, Any
import asyncio
import logging
import json
from datetime import datetime
from sqlalchemy.orm import Session

from .config import Config
from .models import Transaction, get_db

# Set up logging for OpenAI API calls
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a separate logger for OpenAI API interactions
api_logger = logging.getLogger('openai_api')
api_handler = logging.FileHandler('openai_api.log')
api_handler.setLevel(logging.INFO)
api_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
api_handler.setFormatter(api_formatter)
api_logger.addHandler(api_handler)
api_logger.setLevel(logging.INFO)

class EmbeddingService:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        request_data = {
            "model": "text-embedding-3-small",
            "input": text
        }
        
        # Log the request
        api_logger.info(f"EMBEDDING_REQUEST: {json.dumps({
            'timestamp': datetime.now().isoformat(),
            'method': 'create_embedding',
            'model': request_data['model'],
            'input_length': len(text),
            'input_text': text[:200] + '...' if len(text) > 200 else text
        })}")
        
        try:
            response = self.client.embeddings.create(**request_data)
            
            # Log the response
            embedding = response.data[0].embedding
            api_logger.info(f"EMBEDDING_RESPONSE: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'create_embedding',
                'success': True,
                'embedding_dimension': len(embedding),
                'embedding_preview': embedding[:5],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 'unknown',
                    'total_tokens': response.usage.total_tokens if response.usage else 'unknown'
                }
            })}")
            
            return embedding
            
        except Exception as e:
            # Log the error
            api_logger.error(f"EMBEDDING_ERROR: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'create_embedding',
                'success': False,
                'error': str(e),
                'input_text': text[:200] + '...' if len(text) > 200 else text
            })}")
            
            logger.error(f"Error creating embedding: {e}")
            return [0.0] * Config.EMBEDDING_DIMENSION
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        request_data = {
            "model": "text-embedding-3-small",
            "input": texts
        }
        
        # Log the request
        api_logger.info(f"EMBEDDING_BATCH_REQUEST: {json.dumps({
            'timestamp': datetime.now().isoformat(),
            'method': 'create_embeddings_batch',
            'model': request_data['model'],
            'batch_size': len(texts),
            'total_input_length': sum(len(text) for text in texts),
            'sample_inputs': [text[:100] + '...' if len(text) > 100 else text for text in texts[:3]]
        })}")
        
        try:
            response = self.client.embeddings.create(**request_data)
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Log the response
            api_logger.info(f"EMBEDDING_BATCH_RESPONSE: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'create_embeddings_batch',
                'success': True,
                'batch_size': len(embeddings),
                'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 'unknown',
                    'total_tokens': response.usage.total_tokens if response.usage else 'unknown'
                }
            })}")
            
            return embeddings
            
        except Exception as e:
            # Log the error
            api_logger.error(f"EMBEDDING_BATCH_ERROR: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'create_embeddings_batch',
                'success': False,
                'error': str(e),
                'batch_size': len(texts),
                'sample_inputs': [text[:100] + '...' if len(text) > 100 else text for text in texts[:3]]
            })}")
            
            logger.error(f"Error creating batch embeddings: {e}")
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
            
        return float(dot_product / (norm1 * norm2))
    
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
    
    def repopulate_embeddings(self, db: Session, batch_size: int = 50, 
                             max_retries: int = 3, delay: float = 1.0) -> dict:
        """
        Repopulate embeddings for transactions with missing or zero embeddings.
        Returns statistics about the operation.
        """
        import time
        
        # Find transactions needing embeddings
        null_transactions = db.query(Transaction).filter(
            Transaction.embedding.is_(None)
        ).all()
        
        # Find transactions with zero embeddings
        zero_transactions = []
        embedded_transactions = db.query(Transaction).filter(
            Transaction.embedding.isnot(None)
        ).all()
        
        for t in embedded_transactions:
            if t.embedding and all(x == 0.0 for x in t.embedding):
                zero_transactions.append(t)
        
        all_needing_embeddings = null_transactions + zero_transactions
        
        if not all_needing_embeddings:
            return {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'already_had_embeddings': len(embedded_transactions) - len(zero_transactions)
            }
        
        print(f"Repopulating embeddings for {len(all_needing_embeddings)} transactions")
        print(f"  {len(null_transactions)} with null embeddings")
        print(f"  {len(zero_transactions)} with zero embeddings")
        
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, len(all_needing_embeddings), batch_size):
            batch = all_needing_embeddings[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(all_needing_embeddings) - 1) // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} transactions)")
            
            # Prepare texts
            texts = [self.prepare_transaction_text(t) for t in batch]
            
            # Try with retries
            embeddings = None
            for attempt in range(max_retries):
                try:
                    embeddings = self.create_embeddings_batch(texts)
                    
                    # Validate embeddings
                    if embeddings and len(embeddings) == len(texts):
                        valid_count = sum(1 for emb in embeddings if emb and any(x != 0.0 for x in emb))
                        if valid_count > 0:
                            break
                    
                    # If we get here, embeddings were not valid
                    embeddings = None
                    print(f"  Attempt {attempt + 1}: Got invalid embeddings")
                    
                except Exception as e:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    embeddings = None
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * delay
                    print(f"  Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
            
            # Update database
            if embeddings:
                for transaction, embedding in zip(batch, embeddings):
                    if embedding and any(x != 0.0 for x in embedding):
                        transaction.embedding = embedding
                        successful += 1
                    else:
                        failed += 1
                        print(f"  Skipped transaction {transaction.id} due to zero embedding")
                
                try:
                    db.commit()
                    print(f"  Batch {batch_num} completed successfully")
                except Exception as e:
                    print(f"  Error committing batch {batch_num}: {e}")
                    db.rollback()
                    failed += len(batch)
                    successful -= len(batch)
            else:
                print(f"  Batch {batch_num} completely failed")
                failed += len(batch)
            
            # Small delay between batches
            if i + batch_size < len(all_needing_embeddings):
                time.sleep(delay)
        
        return {
            'total_processed': len(all_needing_embeddings),
            'successful': successful,
            'failed': failed,
            'already_had_embeddings': len(embedded_transactions) - len(zero_transactions)
        }
