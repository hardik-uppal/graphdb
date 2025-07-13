import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from sqlalchemy.orm import Session
from collections import defaultdict, Counter
import re
from dataclasses import dataclass

from .models import Transaction, GraphEdge, TransactionCluster
from .embedding_service import EmbeddingService
from .config import Config

@dataclass
class EnhancedNodeFeatures:
    """Enhanced node features for better graph representation."""
    transaction_id: int
    display_label: str
    category: str
    subcategory: str
    merchant_clean: str
    amount_bucket: str
    time_bucket: str
    spending_pattern: str
    frequency_score: float
    centrality_score: float
    semantic_tags: List[str]
    
@dataclass
class TransactionPattern:
    """Represents a detected transaction pattern."""
    pattern_id: str
    pattern_type: str
    pattern_name: str
    transactions: List[int]
    confidence: float
    description: str
    key_features: Dict[str, Any]

class EnhancedGraphService:
    """Enhanced graph service with better node representations and pattern detection."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.graph = nx.Graph()
        self.node_features = {}
        self.detected_patterns = []
        
        # Merchant categorization mappings
        self.merchant_categories = {
            'grocery': ['walmart', 'kroger', 'safeway', 'whole foods', 'trader joes', 'costco', 'target', 'grocery'],
            'restaurant': ['mcdonalds', 'subway', 'starbucks', 'chipotle', 'restaurant', 'cafe', 'pizza', 'burger'],
            'gas': ['shell', 'exxon', 'chevron', 'bp', 'gas', 'fuel', 'station'],
            'retail': ['amazon', 'ebay', 'best buy', 'home depot', 'lowes', 'macy', 'retail'],
            'healthcare': ['cvs', 'walgreens', 'hospital', 'clinic', 'pharmacy', 'medical', 'health'],
            'entertainment': ['netflix', 'spotify', 'cinema', 'theater', 'entertainment', 'movie'],
            'transportation': ['uber', 'lyft', 'taxi', 'bus', 'train', 'airline', 'parking'],
            'utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'utility'],
            'financial': ['bank', 'atm', 'credit', 'loan', 'interest', 'fee'],
            'subscription': ['netflix', 'spotify', 'subscription', 'monthly', 'annual']
        }
        
    def create_enhanced_node_features(self, transactions: List[Transaction]) -> Dict[int, EnhancedNodeFeatures]:
        """Create enhanced node features with meaningful labels and categories."""
        print("Creating enhanced node features...")
        
        enhanced_features = {}
        
        # First pass: collect merchant and amount statistics
        merchant_stats = self._analyze_merchant_patterns(transactions)
        amount_stats = self._analyze_amount_patterns(transactions)
        
        for transaction in transactions:
            # Create enhanced features
            features = self._create_single_node_features(
                transaction, merchant_stats, amount_stats
            )
            enhanced_features[transaction.id] = features
            
        return enhanced_features
    
    def _create_single_node_features(self, transaction: Transaction, 
                                   merchant_stats: Dict, amount_stats: Dict) -> EnhancedNodeFeatures:
        """Create enhanced features for a single transaction."""
        
        # Clean and categorize merchant
        merchant_clean = self._clean_merchant_name(transaction.merchant or "Unknown")
        category, subcategory = self._categorize_merchant(merchant_clean)
        
        # Create meaningful display label
        amount_str = f"${abs(transaction.amount):.2f}"
        date_str = transaction.date.strftime("%m/%d")
        
        if len(merchant_clean) > 15:
            merchant_display = merchant_clean[:12] + "..."
        else:
            merchant_display = merchant_clean
            
        display_label = f"{merchant_display}\n{amount_str} ({date_str})"
        
        # Amount bucketing
        amount_bucket = self._get_amount_bucket(abs(transaction.amount))
        
        # Time bucketing
        time_bucket = self._get_time_bucket(transaction.date)
        
        # Spending pattern analysis
        spending_pattern = self._analyze_spending_pattern(
            transaction, merchant_stats.get(merchant_clean, {})
        )
        
        # Calculate frequency score
        frequency_score = merchant_stats.get(merchant_clean, {}).get('frequency', 0)
        
        # Generate semantic tags
        semantic_tags = self._generate_semantic_tags(transaction, category)
        
        return EnhancedNodeFeatures(
            transaction_id=transaction.id,
            display_label=display_label,
            category=category,
            subcategory=subcategory,
            merchant_clean=merchant_clean,
            amount_bucket=amount_bucket,
            time_bucket=time_bucket,
            spending_pattern=spending_pattern,
            frequency_score=frequency_score,
            centrality_score=0.0,  # Will be calculated later
            semantic_tags=semantic_tags
        )
    
    def _clean_merchant_name(self, merchant: str) -> str:
        """Clean and normalize merchant names."""
        if not merchant:
            return "Unknown"
            
        # Remove common prefixes/suffixes
        merchant = re.sub(r'^(THE|A|AN)\s+', '', merchant.upper())
        merchant = re.sub(r'\s+(INC|LLC|CORP|LTD|CO)\s*$', '', merchant)
        
        # Remove special characters and extra spaces
        merchant = re.sub(r'[^\w\s]', ' ', merchant)
        merchant = re.sub(r'\s+', ' ', merchant).strip()
        
        # Convert to title case
        return merchant.title()
    
    def _categorize_merchant(self, merchant: str) -> Tuple[str, str]:
        """Categorize merchant based on name patterns."""
        merchant_lower = merchant.lower()
        
        for category, keywords in self.merchant_categories.items():
            for keyword in keywords:
                if keyword in merchant_lower:
                    subcategory = self._get_subcategory(merchant_lower, category)
                    return category.title(), subcategory
        
        return "Other", "General"
    
    def _get_subcategory(self, merchant: str, category: str) -> str:
        """Get more specific subcategory."""
        subcategory_maps = {
            'grocery': {
                'walmart': 'Big Box', 'target': 'Big Box', 'costco': 'Warehouse',
                'whole foods': 'Organic', 'trader joes': 'Specialty'
            },
            'restaurant': {
                'mcdonalds': 'Fast Food', 'subway': 'Fast Food', 'starbucks': 'Coffee',
                'pizza': 'Pizza', 'cafe': 'Coffee'
            },
            'retail': {
                'amazon': 'Online', 'ebay': 'Online', 'best buy': 'Electronics',
                'home depot': 'Home Improvement', 'lowes': 'Home Improvement'
            }
        }
        
        if category in subcategory_maps:
            for keyword, subcat in subcategory_maps[category].items():
                if keyword in merchant:
                    return subcat
        
        return category.title()
    
    def _get_amount_bucket(self, amount: float) -> str:
        """Categorize transaction amount into buckets."""
        if amount < 10:
            return "Micro (<$10)"
        elif amount < 50:
            return "Small ($10-50)"
        elif amount < 200:
            return "Medium ($50-200)"
        elif amount < 500:
            return "Large ($200-500)"
        else:
            return "XLarge (>$500)"
    
    def _get_time_bucket(self, date: datetime) -> str:
        """Categorize transaction time."""
        hour = date.hour
        day_of_week = date.strftime("%A")
        
        if 5 <= hour < 12:
            time_period = "Morning"
        elif 12 <= hour < 17:
            time_period = "Afternoon"
        elif 17 <= hour < 21:
            time_period = "Evening"
        else:
            time_period = "Night"
            
        if day_of_week in ["Saturday", "Sunday"]:
            return f"Weekend {time_period}"
        else:
            return f"Weekday {time_period}"
    
    def _analyze_merchant_patterns(self, transactions: List[Transaction]) -> Dict[str, Dict]:
        """Analyze patterns across merchants."""
        merchant_stats = defaultdict(lambda: {
            'count': 0, 'total_amount': 0, 'amounts': [],
            'dates': [], 'frequency': 0
        })
        
        for transaction in transactions:
            merchant = self._clean_merchant_name(transaction.merchant or "Unknown")
            stats = merchant_stats[merchant]
            stats['count'] += 1
            stats['total_amount'] += abs(transaction.amount)
            stats['amounts'].append(abs(transaction.amount))
            stats['dates'].append(transaction.date)
        
        # Calculate frequency scores
        total_transactions = len(transactions)
        for merchant, stats in merchant_stats.items():
            stats['frequency'] = stats['count'] / total_transactions
            stats['avg_amount'] = stats['total_amount'] / stats['count']
            
            # Calculate regularity (how evenly spaced are the transactions)
            if len(stats['dates']) > 1:
                dates_sorted = sorted(stats['dates'])
                intervals = [(dates_sorted[i+1] - dates_sorted[i]).days 
                           for i in range(len(dates_sorted)-1)]
                stats['regularity'] = 1.0 / (1.0 + np.std(intervals)) if intervals else 0.0
            else:
                stats['regularity'] = 0.0
        
        return dict(merchant_stats)
    
    def _analyze_amount_patterns(self, transactions: List[Transaction]) -> Dict:
        """Analyze spending amount patterns."""
        amounts = [abs(t.amount) for t in transactions]
        
        return {
            'mean': np.mean(amounts),
            'median': np.median(amounts),
            'std': np.std(amounts),
            'percentiles': {
                '25': np.percentile(amounts, 25),
                '75': np.percentile(amounts, 75),
                '90': np.percentile(amounts, 90),
                '95': np.percentile(amounts, 95)
            }
        }
    
    def _analyze_spending_pattern(self, transaction: Transaction, merchant_stats: Dict) -> str:
        """Analyze individual transaction spending pattern."""
        amount = abs(transaction.amount)
        
        if not merchant_stats:
            return "Isolated"
            
        avg_amount = merchant_stats.get('avg_amount', amount)
        frequency = merchant_stats.get('frequency', 0)
        regularity = merchant_stats.get('regularity', 0)
        
        # Classify pattern
        if frequency > 0.1:  # More than 10% of all transactions
            if regularity > 0.5:
                return "Regular/Recurring"
            else:
                return "Frequent/Irregular"
        elif frequency > 0.05:
            return "Occasional"
        else:
            if amount > avg_amount * 2:
                return "Large/Unusual"
            else:
                return "Infrequent"
    
    def _generate_semantic_tags(self, transaction: Transaction, category: str) -> List[str]:
        """Generate semantic tags for the transaction."""
        tags = [category.lower()]
        
        amount = abs(transaction.amount)
        
        # Amount-based tags
        if amount > 1000:
            tags.append("high-value")
        elif amount < 10:
            tags.append("micro-transaction")
            
        # Time-based tags
        if transaction.date.weekday() >= 5:  # Weekend
            tags.append("weekend")
        else:
            tags.append("weekday")
            
        # Transaction type tags
        if transaction.transaction_type:
            tags.append(transaction.transaction_type.lower().replace(' ', '-'))
            
        # Special patterns
        if transaction.amount < 0:
            tags.append("income")
        else:
            tags.append("expense")
            
        return tags
    
    def detect_transaction_patterns(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect various transaction patterns in the data."""
        print("Detecting transaction patterns...")
        
        patterns = []
        
        # Recurring payment patterns
        patterns.extend(self._detect_recurring_patterns(transactions))
        
        # Spending burst patterns
        patterns.extend(self._detect_spending_bursts(transactions))
        
        # Merchant loyalty patterns
        patterns.extend(self._detect_loyalty_patterns(transactions))
        
        # Amount clustering patterns
        patterns.extend(self._detect_amount_clusters(transactions))
        
        # Time-based patterns
        patterns.extend(self._detect_temporal_patterns(transactions))
        
        self.detected_patterns = patterns
        return patterns
    
    def _detect_recurring_patterns(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect recurring payment patterns."""
        patterns = []
        merchant_groups = defaultdict(list)
        
        for transaction in transactions:
            merchant = self._clean_merchant_name(transaction.merchant or "Unknown")
            merchant_groups[merchant].append(transaction)
        
        for merchant, group in merchant_groups.items():
            if len(group) >= 3:  # At least 3 transactions
                # Sort by date
                group.sort(key=lambda x: x.date)
                
                # Calculate intervals
                intervals = []
                amounts = []
                for i in range(len(group) - 1):
                    interval = (group[i+1].date - group[i].date).days
                    intervals.append(interval)
                    amounts.append(abs(group[i].amount))
                
                # Check for regularity
                if intervals:
                    avg_interval = np.mean(intervals)
                    interval_std = np.std(intervals)
                    amount_std = np.std(amounts) if amounts else 0
                    
                    # Regular if intervals are consistent and amounts are similar
                    if (interval_std < avg_interval * 0.3 and  # Intervals are consistent
                        amount_std < np.mean(amounts) * 0.2 and  # Amounts are similar
                        20 <= avg_interval <= 35):  # Roughly monthly
                        
                        patterns.append(TransactionPattern(
                            pattern_id=f"recurring_{merchant.lower().replace(' ', '_')}",
                            pattern_type="recurring",
                            pattern_name=f"Monthly {merchant} Payments",
                            transactions=[t.id for t in group],
                            confidence=0.9 - (interval_std / avg_interval),
                            description=f"Regular monthly payments to {merchant}",
                            key_features={
                                'merchant': merchant,
                                'avg_interval_days': avg_interval,
                                'avg_amount': np.mean(amounts),
                                'consistency_score': 1.0 - (interval_std / avg_interval)
                            }
                        ))
        
        return patterns
    
    def _detect_spending_bursts(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect unusual spending burst patterns."""
        patterns = []
        
        # Sort transactions by date
        sorted_transactions = sorted(transactions, key=lambda x: x.date)
        
        # Look for periods of high spending activity
        window_size = 7  # 7-day window
        
        for i in range(len(sorted_transactions) - 3):  # At least 4 transactions
            window_transactions = []
            start_date = sorted_transactions[i].date
            
            # Collect transactions in window
            for j in range(i, len(sorted_transactions)):
                if (sorted_transactions[j].date - start_date).days <= window_size:
                    window_transactions.append(sorted_transactions[j])
                else:
                    break
            
            if len(window_transactions) >= 4:  # At least 4 transactions in window
                total_amount = sum(abs(t.amount) for t in window_transactions)
                avg_daily_spending = total_amount / window_size
                
                # Calculate baseline spending
                all_amounts = [abs(t.amount) for t in transactions]
                baseline_daily = np.mean(all_amounts) * (len(transactions) / 365)  # Rough daily average
                
                # If spending is significantly higher than baseline
                if avg_daily_spending > baseline_daily * 2:
                    patterns.append(TransactionPattern(
                        pattern_id=f"burst_{start_date.strftime('%Y%m%d')}",
                        pattern_type="spending_burst",
                        pattern_name=f"Spending Burst ({start_date.strftime('%m/%d')})",
                        transactions=[t.id for t in window_transactions],
                        confidence=min(0.9, avg_daily_spending / (baseline_daily * 2)),
                        description=f"High spending activity period starting {start_date.strftime('%m/%d/%Y')}",
                        key_features={
                            'total_amount': total_amount,
                            'transaction_count': len(window_transactions),
                            'avg_daily_spending': avg_daily_spending,
                            'burst_ratio': avg_daily_spending / baseline_daily
                        }
                    ))
        
        return patterns
    
    def _detect_loyalty_patterns(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect merchant loyalty patterns."""
        patterns = []
        merchant_stats = self._analyze_merchant_patterns(transactions)
        
        total_amount = sum(abs(t.amount) for t in transactions)
        
        for merchant, stats in merchant_stats.items():
            if stats['count'] >= 5:  # At least 5 transactions
                spending_share = stats['total_amount'] / total_amount
                
                if spending_share > 0.2:  # More than 20% of total spending
                    patterns.append(TransactionPattern(
                        pattern_id=f"loyalty_{merchant.lower().replace(' ', '_')}",
                        pattern_type="loyalty",
                        pattern_name=f"{merchant} Loyalty",
                        transactions=[t.id for t in transactions 
                                    if self._clean_merchant_name(t.merchant or "") == merchant],
                        confidence=min(0.9, spending_share * 2),
                        description=f"High loyalty to {merchant} (${stats['total_amount']:.2f}, {stats['count']} transactions)",
                        key_features={
                            'merchant': merchant,
                            'spending_share': spending_share,
                            'transaction_count': stats['count'],
                            'avg_amount': stats['avg_amount'],
                            'regularity': stats['regularity']
                        }
                    ))
        
        return patterns
    
    def _detect_amount_clusters(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect amount-based clustering patterns."""
        patterns = []
        amounts = [abs(t.amount) for t in transactions]
        
        # Use simple binning to find amount clusters
        hist, bin_edges = np.histogram(amounts, bins=10)
        
        for i, count in enumerate(hist):
            if count >= 3:  # At least 3 transactions in this amount range
                bin_start, bin_end = bin_edges[i], bin_edges[i+1]
                
                # Find transactions in this range
                cluster_transactions = [
                    t for t in transactions 
                    if bin_start <= abs(t.amount) < bin_end
                ]
                
                if len(cluster_transactions) >= 3:
                    patterns.append(TransactionPattern(
                        pattern_id=f"amount_cluster_{i}",
                        pattern_type="amount_cluster",
                        pattern_name=f"${bin_start:.0f}-${bin_end:.0f} Spending",
                        transactions=[t.id for t in cluster_transactions],
                        confidence=min(0.8, count / len(transactions)),
                        description=f"Frequent spending in ${bin_start:.0f}-${bin_end:.0f} range",
                        key_features={
                            'amount_range': [bin_start, bin_end],
                            'transaction_count': len(cluster_transactions),
                            'frequency': count / len(transactions)
                        }
                    ))
        
        return patterns
    
    def _detect_temporal_patterns(self, transactions: List[Transaction]) -> List[TransactionPattern]:
        """Detect time-based patterns."""
        patterns = []
        
        # Weekend vs weekday patterns
        weekend_transactions = [t for t in transactions if t.date.weekday() >= 5]
        weekday_transactions = [t for t in transactions if t.date.weekday() < 5]
        
        if weekend_transactions:
            weekend_amount = sum(abs(t.amount) for t in weekend_transactions)
            total_amount = sum(abs(t.amount) for t in transactions)
            weekend_ratio = weekend_amount / total_amount
            
            if weekend_ratio > 0.4:  # More than 40% spending on weekends
                patterns.append(TransactionPattern(
                    pattern_id="weekend_spender",
                    pattern_type="temporal",
                    pattern_name="Weekend Spender",
                    transactions=[t.id for t in weekend_transactions],
                    confidence=min(0.9, weekend_ratio * 1.5),
                    description=f"High weekend spending activity ({weekend_ratio:.1%} of total)",
                    key_features={
                        'weekend_ratio': weekend_ratio,
                        'weekend_transactions': len(weekend_transactions),
                        'avg_weekend_amount': weekend_amount / len(weekend_transactions)
                    }
                ))
        
        return patterns
    
    def build_enhanced_graph(self, transactions: List[Transaction], db: Session):
        """Build graph with enhanced node features and better representations."""
        print("Building enhanced graph representation...")
        
        # Clear existing graph
        self.graph.clear()
        
        # Create enhanced node features
        self.node_features = self.create_enhanced_node_features(transactions)
        
        # Add nodes with enhanced features
        for transaction in transactions:
            features = self.node_features[transaction.id]
            
            self.graph.add_node(
                transaction.id,
                # Enhanced display properties
                label=features.display_label,
                category=features.category,
                subcategory=features.subcategory,
                merchant=features.merchant_clean,
                amount_bucket=features.amount_bucket,
                time_bucket=features.time_bucket,
                spending_pattern=features.spending_pattern,
                tags=features.semantic_tags,
                
                # Original transaction data
                transaction=transaction,
                amount=transaction.amount,
                date=transaction.date,
                type=transaction.transaction_type,
                
                # Visual properties for graph display
                size=min(50, max(10, abs(transaction.amount) / 10)),  # Node size based on amount
                color=self._get_category_color(features.category),
                shape=self._get_pattern_shape(features.spending_pattern)
            )
        
        # Add enhanced edges
        self._add_enhanced_similarity_edges(transactions, db)
        self._add_enhanced_temporal_edges(transactions, db)
        self._add_enhanced_merchant_edges(transactions, db)
        self._add_pattern_edges(transactions, db)
        
        # Calculate centrality scores and update node features
        self._calculate_and_update_centrality()
        
        print(f"Enhanced graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _get_category_color(self, category: str) -> str:
        """Get color for node based on category."""
        color_map = {
            'Grocery': '#2E8B57',      # Sea Green
            'Restaurant': '#FF6347',    # Tomato
            'Gas': '#FFD700',          # Gold
            'Retail': '#4169E1',       # Royal Blue
            'Healthcare': '#DC143C',    # Crimson
            'Entertainment': '#9932CC', # Dark Orchid
            'Transportation': '#32CD32', # Lime Green
            'Utilities': '#FF8C00',     # Dark Orange
            'Financial': '#8B0000',     # Dark Red
            'Subscription': '#800080',  # Purple
            'Other': '#708090'          # Slate Gray
        }
        return color_map.get(category, '#708090')
    
    def _get_pattern_shape(self, pattern: str) -> str:
        """Get node shape based on spending pattern."""
        shape_map = {
            'Regular/Recurring': 'square',
            'Frequent/Irregular': 'diamond',
            'Occasional': 'circle',
            'Large/Unusual': 'star',
            'Infrequent': 'triangle',
            'Isolated': 'dot'
        }
        return shape_map.get(pattern, 'circle')
    
    def _add_enhanced_similarity_edges(self, transactions: List[Transaction], db: Session):
        """Add semantic similarity edges with enhanced labels."""
        print("Adding enhanced similarity edges...")
        
        for i, trans1 in enumerate(transactions):
            if not trans1.embedding:
                continue
                
            features1 = self.node_features[trans1.id]
            
            for j, trans2 in enumerate(transactions[i+1:], i+1):
                if not trans2.embedding:
                    continue
                
                features2 = self.node_features[trans2.id]
                
                # Calculate semantic similarity
                similarity = self.embedding_service.cosine_similarity(
                    trans1.embedding, trans2.embedding
                )
                
                # Enhanced similarity with category bonus
                if features1.category == features2.category:
                    similarity *= 1.2  # Boost similarity for same category
                
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    # Create descriptive edge label
                    edge_label = f"Similar {features1.category.lower()}"
                    if features1.merchant_clean == features2.merchant_clean:
                        edge_label = f"Same merchant: {features1.merchant_clean}"
                    
                    self.graph.add_edge(
                        trans1.id, trans2.id,
                        weight=similarity,
                        edge_type='similarity',
                        label=edge_label,
                        color='#1f77b4',  # Blue for similarity
                        width=max(1, similarity * 5)
                    )
                    
                    # Store in database
                    edge = GraphEdge(
                        source_transaction_id=trans1.id,
                        target_transaction_id=trans2.id,
                        edge_type='similarity',
                        weight=float(similarity)
                    )
                    db.add(edge)
        
        db.commit()
    
    def _add_enhanced_temporal_edges(self, transactions: List[Transaction], db: Session):
        """Add temporal edges with enhanced context."""
        print("Adding enhanced temporal edges...")
        
        # Sort by date
        sorted_transactions = sorted(transactions, key=lambda x: x.date)
        
        for i in range(len(sorted_transactions) - 1):
            trans1 = sorted_transactions[i]
            trans2 = sorted_transactions[i + 1]
            
            time_diff = abs((trans2.date - trans1.date).days)
            
            # Connect transactions within different time windows
            if time_diff <= 1:
                edge_label = "Same day"
                weight = 1.0
            elif time_diff <= 3:
                edge_label = f"{time_diff} days apart"
                weight = 1.0 / (time_diff + 1)
            elif time_diff <= 7:
                edge_label = "Same week"
                weight = 0.5 / time_diff
            else:
                continue  # Don't connect distant transactions
            
            self.graph.add_edge(
                trans1.id, trans2.id,
                weight=weight,
                edge_type='temporal',
                label=edge_label,
                color='#ff7f0e',  # Orange for temporal
                width=max(1, weight * 3)
            )
            
            edge = GraphEdge(
                source_transaction_id=trans1.id,
                target_transaction_id=trans2.id,
                edge_type='temporal',
                weight=float(weight)
            )
            db.add(edge)
        
        db.commit()
    
    def _add_enhanced_merchant_edges(self, transactions: List[Transaction], db: Session):
        """Add merchant relationship edges."""
        print("Adding enhanced merchant edges...")
        
        merchant_groups = defaultdict(list)
        for transaction in transactions:
            merchant = self.node_features[transaction.id].merchant_clean
            merchant_groups[merchant].append(transaction)
        
        for merchant, group in merchant_groups.items():
            if len(group) > 1:
                # Connect transactions from same merchant
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        trans1, trans2 = group[i], group[j]
                        
                        # Calculate relationship strength
                        features1 = self.node_features[trans1.id]
                        features2 = self.node_features[trans2.id]
                        
                        # Stronger connection for recurring patterns
                        if (features1.spending_pattern == "Regular/Recurring" and 
                            features2.spending_pattern == "Regular/Recurring"):
                            weight = 1.0
                            edge_label = f"Recurring at {merchant}"
                        else:
                            weight = 0.7
                            edge_label = f"Both at {merchant}"
                        
                        self.graph.add_edge(
                            trans1.id, trans2.id,
                            weight=weight,
                            edge_type='merchant',
                            label=edge_label,
                            color='#2ca02c',  # Green for merchant
                            width=max(1, weight * 2)
                        )
                        
                        edge = GraphEdge(
                            source_transaction_id=trans1.id,
                            target_transaction_id=trans2.id,
                            edge_type='merchant',
                            weight=float(weight)
                        )
                        db.add(edge)
        
        db.commit()
    
    def _add_pattern_edges(self, transactions: List[Transaction], db: Session):
        """Add edges based on detected patterns."""
        print("Adding pattern-based edges...")
        
        patterns = self.detect_transaction_patterns(transactions)
        
        for pattern in patterns:
            if len(pattern.transactions) > 1:
                # Connect all transactions in the same pattern
                for i in range(len(pattern.transactions)):
                    for j in range(i + 1, len(pattern.transactions)):
                        trans_id1, trans_id2 = pattern.transactions[i], pattern.transactions[j]
                        
                        if not self.graph.has_edge(trans_id1, trans_id2):
                            self.graph.add_edge(
                                trans_id1, trans_id2,
                                weight=pattern.confidence,
                                edge_type='pattern',
                                pattern_type=pattern.pattern_type,
                                label=f"Part of {pattern.pattern_name}",
                                color='#9467bd',  # Purple for patterns
                                width=max(1, pattern.confidence * 2)
                            )
                            
                            edge = GraphEdge(
                                source_transaction_id=trans_id1,
                                target_transaction_id=trans_id2,
                                edge_type='pattern',
                                weight=float(pattern.confidence)
                            )
                            db.add(edge)
        
        db.commit()
    
    def _calculate_and_update_centrality(self):
        """Calculate centrality measures and update node features."""
        if not self.graph.nodes():
            return
        
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = {}
        
        # Update node features with centrality scores
        for node_id in self.graph.nodes():
            centrality_score = (
                degree_centrality.get(node_id, 0) * 0.4 +
                betweenness_centrality.get(node_id, 0) * 0.4 +
                pagerank.get(node_id, 0) * 0.2
            )
            
            self.node_features[node_id].centrality_score = centrality_score
            
            # Update graph node attributes
            self.graph.nodes[node_id]['centrality_score'] = centrality_score
            
            # Adjust node size based on centrality
            original_size = self.graph.nodes[node_id]['size']
            self.graph.nodes[node_id]['size'] = original_size * (1 + centrality_score)
    
    def get_graph_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about the transaction graph."""
        if not self.graph.nodes():
            return {"error": "No graph data available"}
        
        insights = {
            'basic_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph)
            },
            'category_distribution': self._get_category_distribution(),
            'spending_patterns': self._get_pattern_distribution(),
            'temporal_insights': self._get_temporal_insights(),
            'merchant_insights': self._get_merchant_insights(),
            'detected_patterns': [
                {
                    'pattern_name': p.pattern_name,
                    'pattern_type': p.pattern_type,
                    'confidence': p.confidence,
                    'transaction_count': len(p.transactions),
                    'description': p.description
                }
                for p in self.detected_patterns
            ]
        }
        
        return insights
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of transaction categories."""
        distribution = defaultdict(int)
        for node_id in self.graph.nodes():
            category = self.graph.nodes[node_id]['category']
            distribution[category] += 1
        return dict(distribution)
    
    def _get_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of spending patterns."""
        distribution = defaultdict(int)
        for node_id in self.graph.nodes():
            pattern = self.graph.nodes[node_id]['spending_pattern']
            distribution[pattern] += 1
        return dict(distribution)
    
    def _get_temporal_insights(self) -> Dict[str, Any]:
        """Get temporal spending insights."""
        weekend_count = 0
        weekday_count = 0
        
        for node_id in self.graph.nodes():
            time_bucket = self.graph.nodes[node_id]['time_bucket']
            if 'Weekend' in time_bucket:
                weekend_count += 1
            else:
                weekday_count += 1
        
        return {
            'weekend_transactions': weekend_count,
            'weekday_transactions': weekday_count,
            'weekend_ratio': weekend_count / (weekend_count + weekday_count) if (weekend_count + weekday_count) > 0 else 0
        }
    
    def _get_merchant_insights(self) -> Dict[str, Any]:
        """Get merchant-related insights."""
        merchant_counts = defaultdict(int)
        merchant_amounts = defaultdict(float)
        
        for node_id in self.graph.nodes():
            merchant = self.graph.nodes[node_id]['merchant']
            amount = abs(self.graph.nodes[node_id]['amount'])
            
            merchant_counts[merchant] += 1
            merchant_amounts[merchant] += amount
        
        # Top merchants by frequency and amount
        top_by_frequency = sorted(merchant_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_by_amount = sorted(merchant_amounts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'unique_merchants': len(merchant_counts),
            'top_by_frequency': top_by_frequency,
            'top_by_amount': [(merchant, amount) for merchant, amount in top_by_amount]
        }
