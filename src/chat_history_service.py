"""
Chat History Service for managing chat threads and messages.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .models import ChatThread, ChatMessage
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ChatHistoryService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def create_thread(self, db: Session, title: str = None) -> ChatThread:
        """Create a new chat thread."""
        
        # Auto-generate title if not provided
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        thread = ChatThread(thread_title=title)
        db.add(thread)
        db.commit()
        db.refresh(thread)
        
        return thread
    
    def add_message(self, db: Session, thread_id: int, message_type: str, 
                   content: str, response_time: float = None, 
                   token_usage: int = None) -> ChatMessage:
        """Add a message to a chat thread."""
        
        # Create embedding for the message
        embedding = self.embedding_service.create_embedding(content)
        
        message = ChatMessage(
            thread_id=thread_id,
            message_type=message_type,
            content=content,
            embedding=embedding,
            response_time=response_time,
            token_usage=token_usage
        )
        
        db.add(message)
        
        # Update thread's updated_at
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            thread.updated_at = datetime.utcnow()
            
            # Update thread title if it's the first user message
            messages_count = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread_id
            ).count()
            
            if messages_count == 0 and message_type == 'user':
                # Generate a better title from the first user message
                thread.thread_title = self._generate_thread_title(content)
        
        db.commit()
        db.refresh(message)
        
        return message
    
    def get_thread_messages(self, db: Session, thread_id: int) -> List[ChatMessage]:
        """Get all messages in a thread."""
        return db.query(ChatMessage).filter(
            ChatMessage.thread_id == thread_id
        ).order_by(ChatMessage.created_at).all()
    
    def get_messages(self, db: Session, thread_id: int) -> List[ChatMessage]:
        """Get all messages in a thread (alias for get_thread_messages)."""
        return self.get_thread_messages(db, thread_id)
    
    def get_recent_threads(self, db: Session, limit: int = 20) -> List[ChatThread]:
        """Get recent chat threads."""
        return db.query(ChatThread).order_by(
            desc(ChatThread.updated_at)
        ).limit(limit).all()
    
    def search_messages(self, db: Session, query: str, limit: int = 10) -> List[Tuple[ChatMessage, float]]:
        """Search messages using semantic similarity."""
        
        search_embedding = self.embedding_service.create_embedding(query)
        
        messages = db.query(ChatMessage).filter(
            ChatMessage.embedding.isnot(None)
        ).all()
        
        results = []
        for message in messages:
            if message.embedding:
                similarity = float(self.embedding_service.cosine_similarity(
                    search_embedding, message.embedding
                ))
                results.append((message, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_thread_by_id(self, db: Session, thread_id: int) -> Optional[ChatThread]:
        """Get thread by ID."""
        return db.query(ChatThread).filter(ChatThread.id == thread_id).first()
    
    def delete_thread(self, db: Session, thread_id: int) -> bool:
        """Delete a chat thread and all its messages."""
        
        # Delete all messages in the thread
        db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).delete()
        
        # Delete the thread
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            db.delete(thread)
            db.commit()
            return True
        
        return False
    
    def update_thread_title(self, db: Session, thread_id: int, new_title: str) -> bool:
        """Update thread title."""
        
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            thread.thread_title = new_title
            thread.updated_at = datetime.utcnow()
            db.commit()
            return True
        
        return False
    
    def _generate_thread_title(self, first_message: str) -> str:
        """Generate a meaningful thread title from the first message."""
        
        # Simple title generation - take first few words
        words = first_message.split()
        if len(words) <= 6:
            return first_message
        else:
            return ' '.join(words[:6]) + '...'
    
    def get_chat_analytics(self, db: Session) -> Dict[str, Any]:
        """Get analytics about chat usage."""
        
        total_threads = db.query(ChatThread).count()
        total_messages = db.query(ChatMessage).count()
        
        # Average messages per thread
        avg_messages_per_thread = total_messages / total_threads if total_threads > 0 else 0
        
        # Message types distribution
        user_messages = db.query(ChatMessage).filter(
            ChatMessage.message_type == 'user'
        ).count()
        
        assistant_messages = db.query(ChatMessage).filter(
            ChatMessage.message_type == 'assistant'
        ).count()
        
        return {
            'total_threads': total_threads,
            'total_messages': total_messages,
            'avg_messages_per_thread': avg_messages_per_thread,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages
        }
