"""
Thought Journal Service for handling text, audio, and image entries
with semantic similarity linking to transactions.
"""

import os
import base64
import io
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import ThoughtJournal, Transaction, JournalTransactionLink
from .embedding_service import EmbeddingService
from .config import Config

# Set up logging
logger = logging.getLogger(__name__)

class ThoughtJournalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        
    def create_text_entry(self, db: Session, title: str, content: str, 
                         tags: List[str] = None, mood_score: float = None,
                         importance_score: float = None) -> ThoughtJournal:
        """Create a new text journal entry."""
        
        # Create embedding for the content
        embedding_text = f"Title: {title}\nContent: {content}"
        embedding = self.embedding_service.create_embedding(embedding_text)
        
        # Create journal entry
        journal = ThoughtJournal(
            title=title,
            content=content,
            entry_type='text',
            embedding=embedding,
            mood_score=mood_score,
            importance_score=importance_score,
            tags=tags or []
        )
        
        db.add(journal)
        db.commit()
        db.refresh(journal)
        
        # Auto-link to similar transactions
        self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def create_audio_entry(self, db: Session, title: str, audio_data: bytes,
                          audio_filename: str, transcript: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create a new audio journal entry."""
        
        # If no transcript provided, we'd typically use speech-to-text here
        # For now, we'll use the provided transcript or empty string
        audio_transcript = transcript or ""
        
        # Create embedding from title and transcript
        embedding_text = f"Title: {title}\nTranscript: {audio_transcript}"
        embedding = self.embedding_service.create_embedding(embedding_text)
        
        # Calculate audio duration (simplified - in real implementation you'd use audio library)
        audio_duration = len(audio_data) / 16000  # Rough estimate
        
        # Create journal entry
        journal = ThoughtJournal(
            title=title,
            content=audio_transcript,
            entry_type='audio',
            audio_data=audio_data,
            audio_transcript=audio_transcript,
            audio_filename=audio_filename,
            audio_duration=audio_duration,
            embedding=embedding,
            mood_score=mood_score,
            importance_score=importance_score,
            tags=tags or []
        )
        
        db.add(journal)
        db.commit()
        db.refresh(journal)
        
        # Auto-link to similar transactions
        self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def create_image_entry(self, db: Session, title: str, image_data: bytes,
                          image_filename: str, description: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create a new image journal entry."""
        
        # Get image dimensions
        try:
            image = Image.open(io.BytesIO(image_data))
            image_size = f"{image.width}x{image.height}"
        except Exception:
            image_size = "unknown"
        
        # Use provided description or empty string
        # In real implementation, you'd use image-to-text AI here
        image_description = description or ""
        
        # Create embedding from title and description
        embedding_text = f"Title: {title}\nDescription: {image_description}"
        embedding = self.embedding_service.create_embedding(embedding_text)
        
        # Create journal entry
        journal = ThoughtJournal(
            title=title,
            content=image_description,
            entry_type='image',
            image_data=image_data,
            image_description=image_description,
            image_filename=image_filename,
            image_size=image_size,
            embedding=embedding,
            mood_score=mood_score,
            importance_score=importance_score,
            tags=tags or []
        )
        
        db.add(journal)
        db.commit()
        db.refresh(journal)
        
        # Auto-link to similar transactions
        self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def create_mixed_entry(self, db: Session, title: str, content: str = None,
                          audio_data: bytes = None, audio_filename: str = None,
                          audio_transcript: str = None, image_data: bytes = None,
                          image_filename: str = None, image_description: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create a mixed journal entry with text, audio, and/or image."""
        
        # Combine all text content for embedding
        embedding_parts = [f"Title: {title}"]
        if content:
            embedding_parts.append(f"Content: {content}")
        if audio_transcript:
            embedding_parts.append(f"Audio Transcript: {audio_transcript}")
        if image_description:
            embedding_parts.append(f"Image Description: {image_description}")
        
        embedding_text = "\n".join(embedding_parts)
        embedding = self.embedding_service.create_embedding(embedding_text)
        
        # Process audio
        audio_duration = None
        if audio_data:
            audio_duration = len(audio_data) / 16000  # Rough estimate
        
        # Process image
        image_size = None
        if image_data:
            try:
                image = Image.open(io.BytesIO(image_data))
                image_size = f"{image.width}x{image.height}"
            except Exception:
                image_size = "unknown"
        
        # Create journal entry
        journal = ThoughtJournal(
            title=title,
            content=content,
            entry_type='mixed',
            audio_data=audio_data,
            audio_transcript=audio_transcript,
            audio_filename=audio_filename,
            audio_duration=audio_duration,
            image_data=image_data,
            image_description=image_description,
            image_filename=image_filename,
            image_size=image_size,
            embedding=embedding,
            mood_score=mood_score,
            importance_score=importance_score,
            tags=tags or []
        )
        
        db.add(journal)
        db.commit()
        db.refresh(journal)
        
        # Auto-link to similar transactions
        self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def _auto_link_to_transactions(self, db: Session, journal: ThoughtJournal,
                                  similarity_threshold: float = 0.7):
        """Automatically link journal entry to similar transactions."""
        
        if not journal.embedding:
            return
        
        # Find transactions with similar embeddings
        transactions = db.query(Transaction).filter(
            Transaction.embedding.isnot(None)
        ).all()
        
        auto_linked_ids = []
        links_to_create = []
        
        for transaction in transactions:
            if transaction.embedding:
                similarity = float(self.embedding_service.cosine_similarity(
                    journal.embedding, transaction.embedding
                ))
                
                if similarity >= similarity_threshold:
                    auto_linked_ids.append(transaction.id)
                    
                    # Create link record
                    link = JournalTransactionLink(
                        journal_id=journal.id,
                        transaction_id=transaction.id,
                        similarity_score=similarity,
                        link_type='auto',
                        link_reason=f"Semantic similarity: {similarity:.3f}"
                    )
                    links_to_create.append(link)
        
        # Update journal with auto-linked transactions
        journal.auto_linked_transaction_ids = auto_linked_ids
        
        # Create link records
        for link in links_to_create:
            db.add(link)
        
        db.commit()
    
    def manually_link_to_transaction(self, db: Session, journal_id: int, 
                                   transaction_id: int, reason: str = None):
        """Manually link a journal entry to a transaction."""
        
        journal = db.query(ThoughtJournal).filter(
            ThoughtJournal.id == journal_id
        ).first()
        
        if not journal:
            raise ValueError(f"Journal entry {journal_id} not found")
        
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        # Check if already linked
        existing_link = db.query(JournalTransactionLink).filter(
            JournalTransactionLink.journal_id == journal_id,
            JournalTransactionLink.transaction_id == transaction_id
        ).first()
        
        if existing_link:
            return existing_link
        
        # Calculate similarity if both have embeddings
        similarity_score = 0.0
        if journal.embedding and transaction.embedding:
            similarity_score = float(self.embedding_service.cosine_similarity(
                journal.embedding, transaction.embedding
            ))
        
        # Create link
        link = JournalTransactionLink(
            journal_id=journal_id,
            transaction_id=transaction_id,
            similarity_score=similarity_score,
            link_type='manual',
            link_reason=reason or "Manually linked by user"
        )
        
        db.add(link)
        
        # Update journal's linked transactions
        if not journal.linked_transaction_ids:
            journal.linked_transaction_ids = []
        
        if transaction_id not in journal.linked_transaction_ids:
            journal.linked_transaction_ids.append(transaction_id)
        
        db.commit()
        return link
    
    def get_journal_entries(self, db: Session, limit: int = 50, 
                           entry_type: str = None, tags: List[str] = None,
                           start_date: datetime = None, end_date: datetime = None) -> List[ThoughtJournal]:
        """Get journal entries with optional filters."""
        
        query = db.query(ThoughtJournal)
        
        if entry_type:
            query = query.filter(ThoughtJournal.entry_type == entry_type)
        
        if tags:
            for tag in tags:
                query = query.filter(ThoughtJournal.tags.contains([tag]))
        
        if start_date:
            query = query.filter(ThoughtJournal.created_at >= start_date)
        
        if end_date:
            query = query.filter(ThoughtJournal.created_at <= end_date)
        
        return query.order_by(desc(ThoughtJournal.created_at)).limit(limit).all()
    
    def search_journal_entries(self, db: Session, query_text: str, 
                             limit: int = 10) -> List[Tuple[ThoughtJournal, float]]:
        """Search journal entries using semantic similarity."""
        
        search_embedding = self.embedding_service.create_embedding(query_text)
        
        journal_entries = db.query(ThoughtJournal).filter(
            ThoughtJournal.embedding.isnot(None)
        ).all()
        
        results = []
        for journal in journal_entries:
            if journal.embedding:
                similarity = float(self.embedding_service.cosine_similarity(
                    search_embedding, journal.embedding
                ))
                results.append((journal, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_linked_transactions(self, db: Session, journal_id: int) -> List[Dict[str, Any]]:
        """Get all transactions linked to a journal entry."""
        
        # Get manual and auto links
        links = db.query(JournalTransactionLink).filter(
            JournalTransactionLink.journal_id == journal_id
        ).all()
        
        results = []
        for link in links:
            transaction = db.query(Transaction).filter(
                Transaction.id == link.transaction_id
            ).first()
            
            if transaction:
                results.append({
                    'transaction': transaction,
                    'link': link,
                    'similarity_score': link.similarity_score,
                    'link_type': link.link_type,
                    'link_reason': link.link_reason
                })
        
        return results
    
    def get_journal_analytics(self, db: Session) -> Dict[str, Any]:
        """Get analytics about journal entries."""
        
        total_entries = db.query(ThoughtJournal).count()
        
        # Count by type
        type_counts = db.query(
            ThoughtJournal.entry_type,
            func.count(ThoughtJournal.id).label('count')
        ).group_by(ThoughtJournal.entry_type).all()
        
        # Average mood score
        avg_mood = db.query(
            func.avg(ThoughtJournal.mood_score)
        ).filter(ThoughtJournal.mood_score.isnot(None)).scalar()
        
        # Most common tags
        all_tags = db.query(ThoughtJournal.tags).filter(
            ThoughtJournal.tags.isnot(None)
        ).all()
        
        tag_counts = {}
        for tag_array in all_tags:
            if tag_array[0]:  # tags is an array
                for tag in tag_array[0]:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Total linked transactions
        total_links = db.query(JournalTransactionLink).count()
        
        return {
            'total_entries': total_entries,
            'entries_by_type': {entry_type: count for entry_type, count in type_counts},
            'average_mood': avg_mood,
            'top_tags': top_tags,
            'total_transaction_links': total_links
        }
    
    def update_journal_entry(self, db: Session, journal_id: int, 
                           title: str = None, content: str = None,
                           tags: List[str] = None, mood_score: float = None,
                           importance_score: float = None) -> ThoughtJournal:
        """Update an existing journal entry."""
        
        journal = db.query(ThoughtJournal).filter(
            ThoughtJournal.id == journal_id
        ).first()
        
        if not journal:
            raise ValueError(f"Journal entry {journal_id} not found")
        
        # Update fields
        if title is not None:
            journal.title = title
        if content is not None:
            journal.content = content
        if tags is not None:
            journal.tags = tags
        if mood_score is not None:
            journal.mood_score = mood_score
        if importance_score is not None:
            journal.importance_score = importance_score
        
        # Recalculate embedding if content changed
        if title is not None or content is not None:
            embedding_text = f"Title: {journal.title}\nContent: {journal.content or ''}"
            if journal.audio_transcript:
                embedding_text += f"\nAudio Transcript: {journal.audio_transcript}"
            if journal.image_description:
                embedding_text += f"\nImage Description: {journal.image_description}"
            
            journal.embedding = self.embedding_service.create_embedding(embedding_text)
        
        journal.updated_at = datetime.utcnow()
        db.commit()
        
        # Re-link to transactions if embedding changed
        if title is not None or content is not None:
            self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def delete_journal_entry(self, db: Session, journal_id: int) -> bool:
        """Delete a journal entry and its links."""
        
        journal = db.query(ThoughtJournal).filter(
            ThoughtJournal.id == journal_id
        ).first()
        
        if not journal:
            return False
        
        # Delete all links
        db.query(JournalTransactionLink).filter(
            JournalTransactionLink.journal_id == journal_id
        ).delete()
        
        # Delete journal entry
        db.delete(journal)
        db.commit()
        
        return True
