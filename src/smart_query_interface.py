"""
Smart Query Interface that uses semantic search and context-aware responses.
No rigid function calling - handles any query intelligently.
"""

import openai
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc, and_, or_
from datetime import datetime, timedelta
import re
from collections import defaultdict

from .config import Config
from .models import Transaction, TransactionCluster
from .embedding_service import EmbeddingService
from .graph_service import GraphService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API logger
api_logger = logging.getLogger('openai_api')
api_handler = logging.FileHandler('openai_api.log')
api_handler.setLevel(logging.INFO)
api_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
api_handler.setFormatter(api_formatter)
api_logger.addHandler(api_handler)
api_logger.setLevel(logging.INFO)

class SmartQueryInterface:
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.graph_service = GraphService()
    
    def process_query(self, query: str, db: Session) -> Dict[str, Any]:
        """Process any query intelligently using semantic search and context."""
        print(f"Processing query: {query}")
        
        try:
            # Step 1: Understand the query intent and extract relevant data
            query_context = self._analyze_query_and_get_data(query, db)
            
            # Step 2: Use ChatGPT only for final response generation with the data
            response = self._generate_intelligent_response(query, query_context)
            
            return {
                "query": query,
                "context": query_context,
                "response": response,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "response": "I encountered an error processing your query. Please try again.",
                "success": False
            }
    
    def _analyze_query_and_get_data(self, query: str, db: Session) -> Dict[str, Any]:
        """Analyze the query and gather relevant data using multiple strategies."""
        context = {
            "query_type": "general",
            "time_filters": {},
            "amount_filters": {},
            "merchant_filters": [],
            "category_filters": [],
            "semantic_matches": [],
            "summary_stats": {},
            "relevant_transactions": [],
            "patterns": {},
            "anomalies": []
        }
        
        # Extract time references
        context["time_filters"] = self._extract_time_filters(query)
        
        # Extract amount references
        context["amount_filters"] = self._extract_amount_filters(query)
        
        # Extract merchant/category references
        context["merchant_filters"] = self._extract_merchant_filters(query, db)
        context["category_filters"] = self._extract_category_filters(query, db)
        
        # Semantic search for relevant transactions
        context["semantic_matches"] = self._semantic_search(query, db)
        
        # Get base transaction set
        base_transactions = self._get_filtered_transactions(
            db, context["time_filters"], context["amount_filters"], 
            context["merchant_filters"], context["category_filters"]
        )
        
        # Add semantic matches to relevant transactions
        semantic_transaction_ids = {t.id for t in context["semantic_matches"]}
        base_transaction_ids = {t.id for t in base_transactions}
        
        # Combine both sets
        all_relevant_ids = semantic_transaction_ids.union(base_transaction_ids)
        context["relevant_transactions"] = db.query(Transaction).filter(
            Transaction.id.in_(all_relevant_ids)
        ).all()
        
        # Generate summary statistics
        context["summary_stats"] = self._generate_summary_stats(context["relevant_transactions"])
        
        # Detect patterns and anomalies
        context["patterns"] = self._detect_patterns(context["relevant_transactions"])
        context["anomalies"] = self._detect_anomalies(context["relevant_transactions"])
        
        # Determine query type
        context["query_type"] = self._determine_query_type(query, context)
        
        return context
    
    def _extract_time_filters(self, query: str) -> Dict[str, Any]:
        """Extract time-related filters from the query."""
        time_filters = {}
        query_lower = query.lower()
        
        # Common time patterns
        if "last week" in query_lower or "past week" in query_lower:
            time_filters["start_date"] = datetime.now() - timedelta(days=7)
            time_filters["end_date"] = datetime.now()
        elif "last month" in query_lower or "past month" in query_lower:
            time_filters["start_date"] = datetime.now() - timedelta(days=30)
            time_filters["end_date"] = datetime.now()
        elif "this month" in query_lower:
            today = datetime.now()
            time_filters["start_date"] = today.replace(day=1)
            time_filters["end_date"] = today
        elif "this year" in query_lower:
            today = datetime.now()
            time_filters["start_date"] = today.replace(month=1, day=1)
            time_filters["end_date"] = today
        
        # Extract specific dates (YYYY-MM-DD format)
        date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            time_filters["start_date"] = datetime.strptime(dates[0], "%Y-%m-%d")
            time_filters["end_date"] = datetime.strptime(dates[1], "%Y-%m-%d")
        elif len(dates) == 1:
            time_filters["specific_date"] = datetime.strptime(dates[0], "%Y-%m-%d")
        
        return time_filters
    
    def _extract_amount_filters(self, query: str) -> Dict[str, Any]:
        """Extract amount-related filters from the query."""
        amount_filters = {}
        query_lower = query.lower()
        
        # Extract dollar amounts
        amount_pattern = r'\$(\d+(?:\.\d{2})?)'
        amounts = [float(match) for match in re.findall(amount_pattern, query)]
        
        if amounts:
            if "more than" in query_lower or "greater than" in query_lower or "above" in query_lower:
                amount_filters["min_amount"] = max(amounts)
            elif "less than" in query_lower or "under" in query_lower or "below" in query_lower:
                amount_filters["max_amount"] = min(amounts)
            elif "between" in query_lower and len(amounts) >= 2:
                amount_filters["min_amount"] = min(amounts)
                amount_filters["max_amount"] = max(amounts)
        
        # Detect expense vs income
        if "expense" in query_lower or "spending" in query_lower or "spent" in query_lower:
            amount_filters["type"] = "expense"
        elif "income" in query_lower or "earned" in query_lower or "received" in query_lower:
            amount_filters["type"] = "income"
        
        return amount_filters
    
    def _extract_merchant_filters(self, query: str, db: Session) -> List[str]:
        """Extract merchant names from query using fuzzy matching."""
        merchants = []
        
        # Get all unique merchants
        all_merchants = db.query(Transaction.merchant).filter(
            Transaction.merchant.isnot(None)
        ).distinct().all()
        
        merchant_names = [m[0].lower() for m in all_merchants if m[0]]
        
        # Check for merchant names in query
        query_lower = query.lower()
        for merchant in merchant_names:
            if merchant in query_lower or query_lower in merchant:
                merchants.append(merchant)
        
        return merchants
    
    def _extract_category_filters(self, query: str, db: Session) -> List[str]:
        """Extract transaction categories from query."""
        categories = []
        
        # Get all unique transaction types
        all_types = db.query(Transaction.transaction_type).filter(
            Transaction.transaction_type.isnot(None)
        ).distinct().all()
        
        type_names = [t[0].lower() for t in all_types if t[0]]
        
        # Check for category names in query
        query_lower = query.lower()
        for category in type_names:
            if category in query_lower or query_lower in category:
                categories.append(category)
        
        return categories
    
    def _semantic_search(self, query: str, db: Session, limit: int = 20) -> List[Transaction]:
        """Use semantic search to find relevant transactions."""
        return self.embedding_service.semantic_search(query, db, limit)
    
    def _get_filtered_transactions(self, db: Session, time_filters: Dict, 
                                  amount_filters: Dict, merchant_filters: List,
                                  category_filters: List) -> List[Transaction]:
        """Get transactions based on extracted filters."""
        query = db.query(Transaction)
        
        # Apply time filters
        if "start_date" in time_filters:
            query = query.filter(Transaction.date >= time_filters["start_date"])
        if "end_date" in time_filters:
            query = query.filter(Transaction.date <= time_filters["end_date"])
        if "specific_date" in time_filters:
            query = query.filter(func.date(Transaction.date) == time_filters["specific_date"].date())
        
        # Apply amount filters
        if "min_amount" in amount_filters:
            query = query.filter(Transaction.amount >= amount_filters["min_amount"])
        if "max_amount" in amount_filters:
            query = query.filter(Transaction.amount <= amount_filters["max_amount"])
        if amount_filters.get("type") == "expense":
            query = query.filter(Transaction.amount < 0)
        elif amount_filters.get("type") == "income":
            query = query.filter(Transaction.amount > 0)
        
        # Apply merchant filters
        if merchant_filters:
            merchant_conditions = [
                Transaction.merchant.ilike(f"%{merchant}%") 
                for merchant in merchant_filters
            ]
            query = query.filter(or_(*merchant_conditions))
        
        # Apply category filters
        if category_filters:
            category_conditions = [
                Transaction.transaction_type.ilike(f"%{category}%") 
                for category in category_filters
            ]
            query = query.filter(or_(*category_conditions))
        
        return query.all()
    
    def _generate_summary_stats(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if not transactions:
            return {}
        
        amounts = [t.amount for t in transactions]
        expenses = [t.amount for t in transactions if t.amount < 0]
        income = [t.amount for t in transactions if t.amount > 0]
        
        # Basic stats
        stats = {
            "total_transactions": len(transactions),
            "total_amount": sum(amounts),
            "total_income": sum(income),
            "total_expenses": abs(sum(expenses)),
            "net_amount": sum(amounts),
            "avg_transaction": sum(amounts) / len(amounts) if amounts else 0,
            "date_range": {
                "start": min(t.date for t in transactions),
                "end": max(t.date for t in transactions)
            }
        }
        
        # Top merchants
        merchant_totals = defaultdict(float)
        for t in transactions:
            if t.merchant:
                merchant_totals[t.merchant] += abs(t.amount)
        
        stats["top_merchants"] = sorted(
            merchant_totals.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Top categories
        category_totals = defaultdict(float)
        for t in transactions:
            if t.transaction_type:
                category_totals[t.transaction_type] += abs(t.amount)
        
        stats["top_categories"] = sorted(
            category_totals.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        return stats
    
    def _detect_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect spending patterns and trends."""
        patterns = {}
        
        if not transactions:
            return patterns
        
        # Daily spending pattern
        daily_spending = defaultdict(float)
        for t in transactions:
            day = t.date.strftime('%A')
            daily_spending[day] += abs(t.amount)
        
        patterns["daily_spending"] = dict(daily_spending)
        
        # Monthly trend
        monthly_spending = defaultdict(float)
        for t in transactions:
            month = t.date.strftime('%Y-%m')
            monthly_spending[month] += abs(t.amount)
        
        patterns["monthly_trend"] = dict(monthly_spending)
        
        # Recurring merchants
        merchant_frequency = defaultdict(int)
        for t in transactions:
            if t.merchant:
                merchant_frequency[t.merchant] += 1
        
        patterns["recurring_merchants"] = {
            merchant: count for merchant, count in merchant_frequency.items() 
            if count >= 3
        }
        
        return patterns
    
    def _detect_anomalies(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect anomalous transactions."""
        anomalies = []
        
        if len(transactions) < 10:
            return anomalies
        
        amounts = [abs(t.amount) for t in transactions]
        avg_amount = sum(amounts) / len(amounts)
        
        # Find transactions that are significantly larger than average
        threshold = avg_amount * 3
        
        for t in transactions:
            if abs(t.amount) > threshold:
                anomalies.append({
                    "transaction_id": t.id,
                    "date": t.date.strftime('%Y-%m-%d') if t.date else "Unknown",
                    "amount": t.amount,
                    "transaction_type": t.transaction_type or "Unknown",
                    "merchant": t.merchant or "Unknown",
                    "reason": f"Amount {abs(t.amount):.2f} is {abs(t.amount)/avg_amount:.1f}x larger than average"
                })
        
        return anomalies[:10]  # Limit to top 10
    
    def detect_anomalies(self, db: Session, threshold: float = 90.0) -> List[Dict[str, Any]]:
        """Public method to detect anomalous transactions."""
        transactions = db.query(Transaction).all()
        return self._detect_anomalies(transactions)

    def _determine_query_type(self, query: str, context: Dict[str, Any]) -> str:
        """Determine the type of query for better response generation."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["total", "sum", "how much"]):
            return "summary"
        elif any(word in query_lower for word in ["trend", "pattern", "over time"]):
            return "trend_analysis"
        elif any(word in query_lower for word in ["anomaly", "unusual", "strange"]):
            return "anomaly_detection"
        elif any(word in query_lower for word in ["similar", "like", "compare"]):
            return "similarity_search"
        elif any(word in query_lower for word in ["top", "most", "highest", "largest"]):
            return "ranking"
        else:
            return "general"
    
    def _generate_intelligent_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate intelligent response using ChatGPT with the analyzed data."""
        
        # Prepare context summary for ChatGPT
        context_summary = {
            "query_type": context["query_type"],
            "relevant_transactions_count": len(context["relevant_transactions"]),
            "summary_stats": context["summary_stats"],
            "patterns": context["patterns"],
            "anomalies": context["anomalies"][:3],  # Top 3 anomalies
            "top_transactions": [
                {
                    "date": t.date.strftime("%Y-%m-%d"),
                    "amount": t.amount,
                    "merchant": t.merchant,
                    "type": t.transaction_type
                }
                for t in sorted(context["relevant_transactions"], 
                              key=lambda x: abs(x.amount), reverse=True)[:10]
            ]
        }
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert financial analyst. Based on the user's query and the analyzed transaction data, 
                provide a comprehensive, insightful response. Be specific with numbers and provide actionable insights.
                
                Focus on:
                1. Directly answering the user's question
                2. Providing specific numbers and statistics
                3. Highlighting interesting patterns or anomalies
                4. Offering practical insights or recommendations
                
                Keep your response conversational but professional."""
            },
            {
                "role": "user",
                "content": f"""
                User Query: "{query}"
                
                Analysis Results:
                {json.dumps(context_summary, indent=2, default=str)}
                
                Please provide a comprehensive response to the user's query based on this analysis.
                """
            }
        ]
        
        try:
            # Log the request
            api_logger.info(f"SMART_QUERY_REQUEST: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'context_type': context['query_type'],
                'relevant_transactions': len(context['relevant_transactions'])
            })}")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_content = response.choices[0].message.content
            
            # Log the response
            api_logger.info(f"SMART_QUERY_RESPONSE: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'success': True,
                'response_length': len(response_content),
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 'unknown',
                    'completion_tokens': response.usage.completion_tokens if response.usage else 'unknown',
                    'total_tokens': response.usage.total_tokens if response.usage else 'unknown'
                }
            })}")
            
            return response_content
            
        except Exception as e:
            api_logger.error(f"SMART_QUERY_ERROR: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'error': str(e)
            })}")
            
            # Fallback to data-driven response
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a fallback response when ChatGPT fails."""
        stats = context["summary_stats"]
        
        if not stats:
            return "I couldn't find any relevant transactions for your query."
        
        response = f"Based on your query '{query}', I found {stats['total_transactions']} relevant transactions.\n\n"
        
        if stats.get("total_income", 0) > 0:
            response += f"💰 Total Income: ${stats['total_income']:,.2f}\n"
        
        if stats.get("total_expenses", 0) > 0:
            response += f"💸 Total Expenses: ${stats['total_expenses']:,.2f}\n"
        
        response += f"📊 Net Amount: ${stats['net_amount']:,.2f}\n"
        
        if stats.get("top_merchants"):
            response += f"\n🏪 Top Merchants:\n"
            for merchant, amount in stats["top_merchants"][:3]:
                response += f"  • {merchant}: ${amount:,.2f}\n"
        
        if context["anomalies"]:
            response += f"\n⚠️ Unusual transactions found: {len(context['anomalies'])}\n"
        
        return response
