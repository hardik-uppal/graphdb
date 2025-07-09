"""
Mock query interface for demo when OpenAI quota is exceeded.
"""

from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime

from .models import Transaction

class MockQueryInterface:
    """Mock interface that works without OpenAI API calls."""
    
    def process_natural_language_query(self, query: str, db: Session) -> Dict[str, Any]:
        """Process queries using keyword matching instead of OpenAI."""
        query_lower = query.lower()
        
        # Keyword-based query routing
        if any(word in query_lower for word in ["spend", "spent", "expense", "cost"]):
            if any(word in query_lower for word in ["most", "top", "highest"]):
                return self._get_top_spending_response(db)
            elif any(word in query_lower for word in ["total", "all", "sum"]):
                return self._get_total_spending_response(db)
        
        elif any(word in query_lower for word in ["merchant", "store", "shop"]):
            return self._get_top_merchants_response(db)
        
        elif any(word in query_lower for word in ["monthly", "month", "summary"]):
            return self._get_monthly_summary_response(db)
        
        elif any(word in query_lower for word in ["investment", "invest"]):
            return self._get_investment_transactions_response(db)
        
        elif any(word in query_lower for word in ["category", "type", "breakdown"]):
            return self._get_category_breakdown_response(db)
        
        else:
            return self._get_general_summary_response(db)
    
    def _get_top_spending_response(self, db: Session) -> Dict[str, Any]:
        """Get top spending categories."""
        categories = db.query(
            Transaction.transaction_type,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('count')
        ).filter(
            Transaction.transaction_type.isnot(None),
            Transaction.amount < 0
        ).group_by(Transaction.transaction_type)\
         .order_by(func.sum(Transaction.amount))\
         .limit(5).all()
        
        data = [
            {
                "category": cat.transaction_type,
                "total_spent": abs(cat.total_amount),
                "transaction_count": cat.count
            }
            for cat in categories
        ]
        
        if data:
            top_category = data[0]
            response = f"Your highest spending category is '{top_category['category']}' with ${top_category['total_spent']:,.2f} across {top_category['transaction_count']} transactions."
        else:
            response = "No expense data found."
        
        return {
            "query": "What did I spend the most money on?",
            "function_called": "get_top_spending",
            "data": data,
            "response": response
        }
    
    def _get_total_spending_response(self, db: Session) -> Dict[str, Any]:
        """Get total spending summary."""
        total_expenses = db.query(func.sum(Transaction.amount)).filter(Transaction.amount < 0).scalar() or 0
        total_income = db.query(func.sum(Transaction.amount)).filter(Transaction.amount > 0).scalar() or 0
        transaction_count = db.query(func.count(Transaction.id)).scalar() or 0
        
        data = {
            "total_expenses": abs(total_expenses),
            "total_income": total_income,
            "net_amount": total_income + total_expenses,
            "total_transactions": transaction_count
        }
        
        response = f"Your total expenses are ${abs(total_expenses):,.2f} and total income is ${total_income:,.2f}, giving you a net amount of ${data['net_amount']:,.2f} across {transaction_count} transactions."
        
        return {
            "query": "Total spending summary",
            "function_called": "get_total_spending",
            "data": data,
            "response": response
        }
    
    def _get_top_merchants_response(self, db: Session) -> Dict[str, Any]:
        """Get top merchants by spending."""
        merchants = db.query(
            Transaction.merchant,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('count')
        ).filter(
            Transaction.merchant.isnot(None),
            Transaction.amount < 0
        ).group_by(Transaction.merchant)\
         .order_by(func.sum(Transaction.amount))\
         .limit(5).all()
        
        data = [
            {
                "merchant": merch.merchant.strip(),
                "total_spent": abs(merch.total_amount),
                "transaction_count": merch.count
            }
            for merch in merchants
        ]
        
        if data:
            response = f"Your top merchant by spending is '{data[0]['merchant']}' with ${data[0]['total_spent']:,.2f}."
        else:
            response = "No merchant data found."
        
        return {
            "query": "Top merchants",
            "function_called": "get_top_merchants",
            "data": data,
            "response": response
        }
    
    def _get_monthly_summary_response(self, db: Session) -> Dict[str, Any]:
        """Get current month summary."""
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1)
        
        if now.month == 12:
            end_date = datetime(now.year + 1, 1, 1)
        else:
            end_date = datetime(now.year, now.month + 1, 1)
        
        transactions = db.query(Transaction).filter(
            Transaction.date >= start_date,
            Transaction.date < end_date
        ).all()
        
        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = sum(t.amount for t in transactions if t.amount < 0)
        
        data = {
            "year": now.year,
            "month": now.month,
            "total_transactions": len(transactions),
            "total_income": income,
            "total_expenses": abs(expenses),
            "net_amount": income + expenses
        }
        
        response = f"For {now.strftime('%B %Y')}, you had {len(transactions)} transactions with ${income:,.2f} income and ${abs(expenses):,.2f} expenses, for a net of ${income + expenses:,.2f}."
        
        return {
            "query": "Monthly summary",
            "function_called": "get_monthly_summary",
            "data": data,
            "response": response
        }
    
    def _get_investment_transactions_response(self, db: Session) -> Dict[str, Any]:
        """Get investment-related transactions."""
        investments = db.query(Transaction).filter(
            Transaction.transaction_type.ilike('%investment%')
        ).order_by(desc(Transaction.date)).limit(10).all()
        
        data = [
            {
                "date": t.date.strftime("%Y-%m-%d"),
                "amount": t.amount,
                "merchant": t.merchant,
                "description": t.description
            }
            for t in investments
        ]
        
        total_invested = sum(t.amount for t in investments if t.amount < 0)
        
        response = f"Found {len(investments)} investment transactions. Total invested: ${abs(total_invested):,.2f}."
        
        return {
            "query": "Investment transactions",
            "function_called": "get_investment_transactions",
            "data": data,
            "response": response
        }
    
    def _get_category_breakdown_response(self, db: Session) -> Dict[str, Any]:
        """Get spending breakdown by category."""
        categories = db.query(
            Transaction.transaction_type,
            func.sum(Transaction.amount).label('total_amount'),
            func.count(Transaction.id).label('count')
        ).filter(Transaction.transaction_type.isnot(None))\
         .group_by(Transaction.transaction_type)\
         .order_by(desc(func.abs(func.sum(Transaction.amount))))\
         .all()
        
        data = [
            {
                "category": cat.transaction_type,
                "total_amount": cat.total_amount,
                "transaction_count": cat.count,
                "is_expense": cat.total_amount < 0
            }
            for cat in categories
        ]
        
        response = f"Found {len(data)} transaction categories. Your spending is spread across various categories like {', '.join([d['category'] for d in data[:3]])}."
        
        return {
            "query": "Category breakdown",
            "function_called": "get_category_breakdown",
            "data": data,
            "response": response
        }
    
    def _get_general_summary_response(self, db: Session) -> Dict[str, Any]:
        """Get general account summary."""
        total_transactions = db.query(func.count(Transaction.id)).scalar() or 0
        date_range = db.query(
            func.min(Transaction.date),
            func.max(Transaction.date)
        ).first()
        
        data = {
            "total_transactions": total_transactions,
            "date_range": {
                "start": date_range[0],
                "end": date_range[1]
            }
        }
        
        response = f"Your account has {total_transactions} transactions from {date_range[0].strftime('%Y-%m-%d') if date_range[0] else 'N/A'} to {date_range[1].strftime('%Y-%m-%d') if date_range[1] else 'N/A'}. You can ask me about spending, merchants, categories, or monthly summaries."
        
        return {
            "query": "General summary",
            "function_called": "get_general_summary", 
            "data": data,
            "response": response
        }
