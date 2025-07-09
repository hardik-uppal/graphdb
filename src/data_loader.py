import pandas as pd
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session

from .models import Transaction, get_db
from .embedding_service import EmbeddingService

class DataLoader:
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    def parse_scotiabank_csv(self, csv_path: str) -> pd.DataFrame:
        """Parse Scotiabank CSV file."""
        print(f"Loading data from {csv_path}")
        
        # Read CSV with specific column names
        df = pd.read_csv(
            csv_path,
            header=None,
            names=['date', 'amount', 'unknown', 'transaction_type', 'merchant']
        )
        
        # Clean and process data
        df = self._clean_data(df)
        
        print(f"Loaded {len(df)} transactions")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
        
        # Clean amounts (remove commas, convert to float)
        df['amount'] = df['amount'].astype(str).str.replace(',', '').astype(float)
        
        # Clean text fields
        df['transaction_type'] = df['transaction_type'].astype(str).str.strip()
        df['merchant'] = df['merchant'].astype(str).str.strip()
        
        # Replace empty strings with None
        df['merchant'] = df['merchant'].replace('', None)
        df['transaction_type'] = df['transaction_type'].replace('', None)
        
        # Create description field
        df['description'] = df.apply(self._create_description, axis=1)
        
        # Remove rows with invalid data
        df = df.dropna(subset=['date', 'amount'])
        
        return df
    
    def _create_description(self, row) -> str:
        """Create a description field from available data."""
        parts = []
        
        if pd.notna(row['transaction_type']) and row['transaction_type'] != 'nan':
            parts.append(row['transaction_type'])
        
        if pd.notna(row['merchant']) and row['merchant'] != 'nan':
            parts.append(row['merchant'])
        
        return ' - '.join(parts) if parts else 'Transaction'
    
    def load_to_database(self, df: pd.DataFrame, db: Session) -> List[Transaction]:
        """Load DataFrame to database."""
        print("Loading transactions to database...")
        
        transactions = []
        
        for _, row in df.iterrows():
            transaction = Transaction(
                date=row['date'].to_pydatetime(),
                amount=float(row['amount']),
                transaction_type=row['transaction_type'],
                merchant=row['merchant'],
                description=row['description']
            )
            
            db.add(transaction)
            transactions.append(transaction)
        
        db.commit()
        
        # Refresh to get IDs
        for transaction in transactions:
            db.refresh(transaction)
        
        print(f"Loaded {len(transactions)} transactions to database")
        return transactions
    
    def process_csv_file(self, csv_path: str, db: Session) -> List[Transaction]:
        """Complete processing pipeline for CSV file."""
        # Parse CSV
        df = self.parse_scotiabank_csv(csv_path)
        
        # Load to database
        transactions = self.load_to_database(df, db)
        
        # Create embeddings
        self.embedding_service.update_transaction_embeddings(db)
        
        # Refresh transactions to get embeddings
        for transaction in transactions:
            db.refresh(transaction)
        
        return transactions
    
    def get_summary_statistics(self, transactions: List[Transaction]) -> dict:
        """Get summary statistics for transactions."""
        if not transactions:
            return {}
        
        amounts = [t.amount for t in transactions]
        types = [t.transaction_type for t in transactions if t.transaction_type]
        merchants = [t.merchant for t in transactions if t.merchant]
        
        # Calculate statistics
        total_income = sum(a for a in amounts if a > 0)
        total_expenses = sum(a for a in amounts if a < 0)
        
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        merchant_counts = {}
        for m in merchants:
            merchant_counts[m] = merchant_counts.get(m, 0) + 1
        
        return {
            'total_transactions': len(transactions),
            'total_income': total_income,
            'total_expenses': abs(total_expenses),
            'net_amount': total_income + total_expenses,
            'avg_transaction': sum(amounts) / len(amounts),
            'date_range': {
                'start': min(t.date for t in transactions),
                'end': max(t.date for t in transactions)
            },
            'top_transaction_types': sorted(
                type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'top_merchants': sorted(
                merchant_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
