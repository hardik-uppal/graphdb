import openai
import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from datetime import datetime, timedelta

from .config import Config
from .models import Transaction, TransactionCluster
from .embedding_service import EmbeddingService
from .graph_service import GraphService

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

class QueryInterface:
    def __init__(self):
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        self.embedding_service = EmbeddingService()
        self.graph_service = GraphService()
        
        # Define available functions for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_total_spending",
                    "description": "Get total spending amount for a specific period or category",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                            "transaction_type": {"type": "string", "description": "Transaction type filter"},
                            "merchant": {"type": "string", "description": "Merchant name filter"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_merchants",
                    "description": "Get top merchants by spending amount",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of top merchants to return"},
                            "transaction_type": {"type": "string", "description": "Transaction type filter"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_spending_by_category",
                    "description": "Get spending breakdown by transaction category/type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_similar_transactions",
                    "description": "Find transactions similar to a given description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "Description to search for"},
                            "limit": {"type": "integer", "description": "Number of results to return"}
                        },
                        "required": ["description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_monthly_summary",
                    "description": "Get monthly spending summary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "year": {"type": "integer", "description": "Year"},
                            "month": {"type": "integer", "description": "Month (1-12)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_anomalies",
                    "description": "Detect unusual or anomalous transactions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "threshold": {"type": "number", "description": "Anomaly threshold (0-100)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_transaction_clusters",
                    "description": "Get transaction clusters and patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cluster_count": {"type": "integer", "description": "Number of clusters"}
                        }
                    }
                }
            }
        ]
    
    def process_natural_language_query(self, query: str, db: Session) -> Dict[str, Any]:
        """Process a natural language query and return results."""
        print(f"Processing query: {query}")
        
        try:
            # Prepare the chat completion request
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial data analyst. Given a user query about transaction data, 
                    determine which function to call with appropriate parameters. 
                    Always provide exact parameter values, don't use placeholders."""
                },
                {"role": "user", "content": query}
            ]
            
            # Log the request
            api_logger.info(f"CHAT_COMPLETION_REQUEST: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'process_natural_language_query',
                'model': 'gpt-4',
                'messages': messages,
                'tools_count': len(self.tools),
                'tool_choice': 'auto',
                'user_query': query
            })}")
            
            # Use OpenAI to determine which function to call
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            # Check if the model wants to call a function
            message = response.choices[0].message
            
            # Log the response
            api_logger.info(f"CHAT_COMPLETION_RESPONSE: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'process_natural_language_query',
                'success': True,
                'finish_reason': response.choices[0].finish_reason,
                'has_tool_calls': hasattr(message, 'tool_calls') and message.tool_calls is not None,
                'tool_calls': [
                    {
                        'function_name': tc.function.name,
                        'function_args': tc.function.arguments
                    } for tc in (message.tool_calls or [])
                ],
                'message_content': message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 'unknown',
                    'completion_tokens': response.usage.completion_tokens if response.usage else 'unknown',
                    'total_tokens': response.usage.total_tokens if response.usage else 'unknown'
                }
            })}")
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Calling function: {function_name} with args: {function_args}")
                
                # Call the appropriate function
                result = self._execute_function(function_name, function_args, db)
                
                # Generate natural language response
                response_text = self._generate_response(query, function_name, function_args, result)
                
                return {
                    "query": query,
                    "function_called": function_name,
                    "function_args": function_args,
                    "data": result,
                    "response": response_text
                }
            else:
                # No function call needed, generate direct response
                return {
                    "query": query,
                    "response": message.content,
                    "data": None
                }
                
        except Exception as e:
            # Log the error
            api_logger.error(f"CHAT_COMPLETION_ERROR: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': 'process_natural_language_query',
                'success': False,
                'error': str(e),
                'user_query': query
            })}")
            
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "response": "I encountered an error processing your query. Please try rephrasing it."
            }
    
    def _execute_function(self, function_name: str, args: Dict[str, Any], db: Session) -> Any:
        """Execute the specified function with given arguments."""
        if function_name == "get_total_spending":
            return self.get_total_spending(db, **args)
        elif function_name == "get_top_merchants":
            return self.get_top_merchants(db, **args)
        elif function_name == "get_spending_by_category":
            return self.get_spending_by_category(db, **args)
        elif function_name == "find_similar_transactions":
            return self.find_similar_transactions(db, **args)
        elif function_name == "get_monthly_summary":
            return self.get_monthly_summary(db, **args)
        elif function_name == "detect_anomalies":
            return self.detect_anomalies(db, **args)
        elif function_name == "get_transaction_clusters":
            return self.get_transaction_clusters(db, **args)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    def get_total_spending(self, db: Session, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, transaction_type: Optional[str] = None,
                          merchant: Optional[str] = None) -> Dict[str, Any]:
        """Get total spending with optional filters."""
        query = db.query(Transaction)
        
        # Apply filters
        if start_date:
            query = query.filter(Transaction.date >= datetime.strptime(start_date, "%Y-%m-%d"))
        if end_date:
            query = query.filter(Transaction.date <= datetime.strptime(end_date, "%Y-%m-%d"))
        if transaction_type:
            query = query.filter(Transaction.transaction_type.ilike(f"%{transaction_type}%"))
        if merchant:
            query = query.filter(Transaction.merchant.ilike(f"%{merchant}%"))
        
        transactions = query.all()
        
        total_amount = sum(t.amount for t in transactions)
        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = sum(t.amount for t in transactions if t.amount < 0)
        
        return {
            "total_transactions": len(transactions),
            "total_amount": total_amount,
            "total_income": income,
            "total_expenses": abs(expenses),
            "net_amount": total_amount,
            "filters_applied": {
                "start_date": start_date,
                "end_date": end_date,
                "transaction_type": transaction_type,
                "merchant": merchant
            }
        }
    
    def get_top_merchants(self, db: Session, limit: int = 10, 
                         transaction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get top merchants by spending amount."""
        query = db.query(
            Transaction.merchant,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('transaction_count')
        ).filter(Transaction.merchant.isnot(None))
        
        if transaction_type:
            query = query.filter(Transaction.transaction_type.ilike(f"%{transaction_type}%"))
        
        # Group by merchant and order by absolute amount (expenses)
        results = query.group_by(Transaction.merchant)\
                      .having(func.sum(Transaction.amount) < 0)\
                      .order_by(func.sum(Transaction.amount))\
                      .limit(limit).all()
        
        return [
            {
                "merchant": result.merchant,
                "total_spent": abs(result.total_amount),
                "transaction_count": result.transaction_count
            }
            for result in results
        ]
    
    def get_spending_by_category(self, db: Session, start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get spending breakdown by category."""
        query = db.query(
            Transaction.transaction_type,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('transaction_count')
        ).filter(Transaction.transaction_type.isnot(None))
        
        if start_date:
            query = query.filter(Transaction.date >= datetime.strptime(start_date, "%Y-%m-%d"))
        if end_date:
            query = query.filter(Transaction.date <= datetime.strptime(end_date, "%Y-%m-%d"))
        
        results = query.group_by(Transaction.transaction_type)\
                      .order_by(desc(func.abs(func.sum(Transaction.amount))))\
                      .all()
        
        return [
            {
                "category": result.transaction_type,
                "total_amount": result.total_amount,
                "transaction_count": result.transaction_count,
                "is_expense": result.total_amount < 0
            }
            for result in results
        ]
    
    def find_similar_transactions(self, db: Session, description: str, 
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Find transactions similar to given description."""
        similar_transactions = self.embedding_service.semantic_search(
            description, db, limit
        )
        
        return [
            {
                "id": t.id,
                "date": t.date.strftime("%Y-%m-%d"),
                "amount": t.amount,
                "transaction_type": t.transaction_type,
                "merchant": t.merchant,
                "description": t.description
            }
            for t in similar_transactions
        ]
    
    def get_monthly_summary(self, db: Session, year: Optional[int] = None, 
                           month: Optional[int] = None) -> Dict[str, Any]:
        """Get monthly spending summary."""
        if not year or not month:
            # Use current month if not specified
            now = datetime.now()
            year = year or now.year
            month = month or now.month
        
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        
        transactions = db.query(Transaction).filter(
            Transaction.date >= start_date,
            Transaction.date <= end_date
        ).all()
        
        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = sum(t.amount for t in transactions if t.amount < 0)
        
        # Top categories
        categories = {}
        for t in transactions:
            if t.transaction_type:
                categories[t.transaction_type] = categories.get(t.transaction_type, 0) + t.amount
        
        return {
            "year": year,
            "month": month,
            "total_transactions": len(transactions),
            "total_income": income,
            "total_expenses": abs(expenses),
            "net_amount": income + expenses,
            "top_categories": sorted(
                [(k, abs(v)) for k, v in categories.items() if v < 0],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def detect_anomalies(self, db: Session, threshold: float = 95.0) -> List[Dict[str, Any]]:
        """Detect anomalous transactions."""
        transactions = db.query(Transaction).all()
        
        if not transactions:
            return []
        
        # Build graph if not already built
        if not self.graph_service.graph.nodes():
            self.graph_service.build_graph_from_transactions(transactions, db)
        
        anomalous_transactions = self.graph_service.find_anomalous_transactions(
            transactions, threshold
        )
        
        return [
            {
                "id": t.id,
                "date": t.date.strftime("%Y-%m-%d"),
                "amount": t.amount,
                "transaction_type": t.transaction_type,
                "merchant": t.merchant,
                "description": t.description,
                "reason": "Unusual graph connectivity pattern"
            }
            for t in anomalous_transactions
        ]
    
    def get_transaction_clusters(self, db: Session, cluster_count: int = 10) -> List[Dict[str, Any]]:
        """Get transaction clusters."""
        # Check if clusters already exist
        existing_clusters = db.query(TransactionCluster).all()
        
        if not existing_clusters:
            # Create clusters
            transactions = db.query(Transaction).all()
            if transactions:
                self.graph_service.detect_transaction_clusters(
                    transactions, db, cluster_count
                )
                existing_clusters = db.query(TransactionCluster).all()
        
        return [
            {
                "id": cluster.id,
                "name": cluster.name,
                "description": cluster.description,
                "transaction_count": len(cluster.transaction_ids),
                "transaction_ids": cluster.transaction_ids[:10]  # Limit for display
            }
            for cluster in existing_clusters
        ]
    
    def _generate_response(self, query: str, function_name: str, 
                          function_args: Dict[str, Any], result: Any) -> str:
        """Generate natural language response based on function results."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful financial assistant. Based on the user's query and the data provided, 
                    generate a clear, conversational response that summarizes the findings. 
                    Include specific numbers and insights. Be concise but informative."""
                },
                {
                    "role": "user", 
                    "content": f"""
                    User asked: "{query}"
                    Function called: {function_name}
                    Function arguments: {json.dumps(function_args, default=str)}
                    Results: {json.dumps(result, default=str, indent=2)}
                    
                    Please provide a natural language summary of these results.
                    """
                }
            ]
            
            # Log the request
            api_logger.info(f"CHAT_COMPLETION_REQUEST: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': '_generate_response',
                'model': 'gpt-4',
                'messages': messages,
                'max_tokens': 500,
                'user_query': query,
                'function_name': function_name,
                'function_args': function_args
            })}")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500
            )
            
            response_content = response.choices[0].message.content
            
            # Log the response
            api_logger.info(f"CHAT_COMPLETION_RESPONSE: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': '_generate_response',
                'success': True,
                'finish_reason': response.choices[0].finish_reason,
                'response_content': response_content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 'unknown',
                    'completion_tokens': response.usage.completion_tokens if response.usage else 'unknown',
                    'total_tokens': response.usage.total_tokens if response.usage else 'unknown'
                }
            })}")
            
            return response_content
            
        except Exception as e:
            # Log the error
            api_logger.error(f"CHAT_COMPLETION_ERROR: {json.dumps({
                'timestamp': datetime.now().isoformat(),
                'method': '_generate_response',
                'success': False,
                'error': str(e),
                'user_query': query,
                'function_name': function_name
            })}")
            
            logger.error(f"Error generating response: {e}")
            return f"I found the data for your query about {query}, but couldn't generate a proper response. Please check the raw data above."
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I found some information but had trouble explaining it clearly. The function {function_name} returned data that you can review in the detailed results."
