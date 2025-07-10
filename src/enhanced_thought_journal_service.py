"""
Enhanced Thought Journal Service that uses OpenAI to extract metadata.
"""

import os
import base64
import io
import json
import logging
import openai
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from .models import ThoughtJournal, Transaction, JournalTransactionLink
from .embedding_service import EmbeddingService
from .config import Config

logger = logging.getLogger(__name__)

class EnhancedThoughtJournalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def create_journal_entry(self, db: Session, content: str, 
                           audio_data: bytes = None, audio_filename: str = None,
                           image_data: bytes = None, image_filename: str = None) -> ThoughtJournal:
        """Create a journal entry with AI-generated metadata."""
        
        # Extract metadata using OpenAI
        metadata = self._extract_metadata_with_ai(content, audio_data, image_data)
        
        # Determine entry type
        entry_type = 'mixed'
        if audio_data and image_data:
            entry_type = 'mixed'
        elif audio_data:
            entry_type = 'audio'
        elif image_data:
            entry_type = 'image'
        else:
            entry_type = 'text'
        
        # Create embedding for all content
        embedding_text = self._prepare_embedding_text(content, metadata)
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
            title=metadata.get('title', 'Untitled Entry'),
            content=content,
            entry_type=entry_type,
            audio_data=audio_data,
            audio_transcript=metadata.get('audio_transcript', ''),
            audio_filename=audio_filename,
            audio_duration=audio_duration,
            image_data=image_data,
            image_description=metadata.get('image_description', ''),
            image_filename=image_filename,
            image_size=image_size,
            embedding=embedding,
            mood_score=metadata.get('mood_score'),
            importance_score=metadata.get('importance_score'),
            tags=metadata.get('tags', [])
        )
        
        db.add(journal)
        db.commit()
        db.refresh(journal)
        
        # Auto-link to similar transactions
        self._auto_link_to_transactions(db, journal)
        
        return journal
    
    def _extract_metadata_with_ai(self, content: str, audio_data: bytes = None, 
                                 image_data: bytes = None) -> Dict[str, Any]:
        """Use OpenAI to extract metadata from the content."""
        
        # Prepare the prompt
        prompt_parts = [f"Content: {content}"]
        
        if audio_data:
            prompt_parts.append("Note: This entry includes an audio recording.")
        
        if image_data:
            prompt_parts.append("Note: This entry includes an image.")
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI assistant that analyzes journal entries to extract metadata. 
                        Based on the content provided, generate:
                        1. A concise title (max 50 characters)
                        2. Importance score (0.0 to 1.0) - how significant is this entry?
                        3. Mood score (0.0 to 1.0) - emotional tone (0=negative, 0.5=neutral, 1=positive)
                        4. Tags (3-5 relevant tags)
                        5. If audio is mentioned, suggest what the transcript might contain
                        6. If image is mentioned, suggest what the image might show
                        
                        Respond in JSON format with keys: title, importance_score, mood_score, tags, audio_transcript, image_description"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content
            metadata = json.loads(response_text)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata with AI: {e}")
            
            # Fallback to simple metadata extraction
            return {
                'title': content[:50] + '...' if len(content) > 50 else content,
                'importance_score': 0.5,
                'mood_score': 0.5,
                'tags': ['journal', 'entry'],
                'audio_transcript': '',
                'image_description': ''
            }
    
    def _prepare_embedding_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding creation."""
        
        parts = [f"Title: {metadata.get('title', '')}"]
        
        if content:
            parts.append(f"Content: {content}")
        
        if metadata.get('audio_transcript'):
            parts.append(f"Audio: {metadata['audio_transcript']}")
        
        if metadata.get('image_description'):
            parts.append(f"Image: {metadata['image_description']}")
        
        if metadata.get('tags'):
            parts.append(f"Tags: {', '.join(metadata['tags'])}")
        
        return "\n".join(parts)
    
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
    
    def create_mixed_entry(self, db: Session, title: str, content: str = None,
                          audio_data: bytes = None, audio_filename: str = None,
                          audio_transcript: str = None, image_data: bytes = None,
                          image_filename: str = None, image_description: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create a mixed-type journal entry with AI-enhanced metadata."""
        
        # Use AI to extract metadata if not provided
        if not all([tags, mood_score, importance_score]):
            ai_metadata = self._extract_metadata_with_ai(content, audio_data, image_data)
            
            if not title:
                title = ai_metadata.get('title', 'Mixed Entry')
            if tags is None:
                tags = ai_metadata.get('tags', [])
            if mood_score is None:
                mood_score = ai_metadata.get('mood_score', 0.5)
            if importance_score is None:
                importance_score = ai_metadata.get('importance_score', 0.5)
            if not audio_transcript and audio_data:
                audio_transcript = ai_metadata.get('audio_transcript', '')
            if not image_description and image_data:
                image_description = ai_metadata.get('image_description', '')
        
        # Create embedding text
        embedding_text = self._prepare_embedding_text_manual(
            title, content, audio_transcript, image_description, tags
        )
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
    
    def create_text_entry(self, db: Session, title: str, content: str,
                         tags: List[str] = None, mood_score: float = None,
                         importance_score: float = None) -> ThoughtJournal:
        """Create a text-only journal entry."""
        return self.create_mixed_entry(
            db, title, content=content, tags=tags, 
            mood_score=mood_score, importance_score=importance_score
        )
    
    def create_audio_entry(self, db: Session, title: str, audio_data: bytes,
                          audio_filename: str = None, audio_transcript: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create an audio journal entry."""
        return self.create_mixed_entry(
            db, title, audio_data=audio_data, audio_filename=audio_filename,
            audio_transcript=audio_transcript, tags=tags,
            mood_score=mood_score, importance_score=importance_score
        )
    
    def create_image_entry(self, db: Session, title: str, image_data: bytes,
                          image_filename: str = None, image_description: str = None,
                          tags: List[str] = None, mood_score: float = None,
                          importance_score: float = None) -> ThoughtJournal:
        """Create an image journal entry."""
        return self.create_mixed_entry(
            db, title, image_data=image_data, image_filename=image_filename,
            image_description=image_description, tags=tags,
            mood_score=mood_score, importance_score=importance_score
        )
    
    def _prepare_embedding_text_manual(self, title: str, content: str = None,
                                      audio_transcript: str = None,
                                      image_description: str = None,
                                      tags: List[str] = None) -> str:
        """Prepare text for embedding creation from manual parameters."""
        
        parts = [f"Title: {title}"]
        
        if content:
            parts.append(f"Content: {content}")
        
        if audio_transcript:
            parts.append(f"Audio: {audio_transcript}")
        
        if image_description:
            parts.append(f"Image: {image_description}")
        
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        
        return "\n".join(parts)
    
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
        
        # Average importance score
        avg_importance = db.query(
            func.avg(ThoughtJournal.importance_score)
        ).filter(ThoughtJournal.importance_score.isnot(None)).scalar()
        
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
            'average_importance': avg_importance,
            'top_tags': top_tags,
            'total_transaction_links': total_links
        }
    
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
    
    def manually_link_to_transaction(self, db: Session, journal_id: int, transaction_id: int, 
                                    link_reason: str = None) -> bool:
        """Manually link a journal entry to a transaction."""
        
        # Check if link already exists
        existing_link = db.query(JournalTransactionLink).filter(
            JournalTransactionLink.journal_id == journal_id,
            JournalTransactionLink.transaction_id == transaction_id
        ).first()
        
        if existing_link:
            return False  # Link already exists
        
        # Create new link
        link = JournalTransactionLink(
            journal_id=journal_id,
            transaction_id=transaction_id,
            similarity_score=1.0,  # Manual links get max similarity
            link_type='manual',
            link_reason=link_reason or 'Manually linked by user'
        )
        
        db.add(link)
        
        # Update journal's auto_linked_transaction_ids
        journal = db.query(ThoughtJournal).filter(ThoughtJournal.id == journal_id).first()
        if journal:
            if journal.auto_linked_transaction_ids is None:
                journal.auto_linked_transaction_ids = []
            if transaction_id not in journal.auto_linked_transaction_ids:
                journal.auto_linked_transaction_ids.append(transaction_id)
        
        db.commit()
        return True
