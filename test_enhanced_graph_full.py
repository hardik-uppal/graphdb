"""
Test Enhanced Graph Database Functionality
This script tests the enhanced graph service with better node representations and analytics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_db, Transaction
from src.enhanced_graph_service import EnhancedGraphService
from src.enhanced_graph_visualizer import EnhancedGraphVisualizer

def test_enhanced_graph_functionality():
    """Test the enhanced graph functionality with better analytics."""
    
    print("🔬 Testing Enhanced Graph Database Functionality")
    print("=" * 60)
    
    # Initialize services
    print("Initializing enhanced services...")
    graph_service = EnhancedGraphService()
    visualizer = EnhancedGraphVisualizer()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Get transactions
        print("\\n📊 Loading transactions...")
        transactions = db.query(Transaction).limit(100).all()  # Test with more data
        print(f"Found {len(transactions)} transactions")
        
        if not transactions:
            print("❌ No transactions found in database!")
            return
        
        # Test enhanced node features
        print("\\n🏷️  Creating enhanced node features...")
        enhanced_features = graph_service.create_enhanced_node_features(transactions)
        print(f"✅ Created {len(enhanced_features)} enhanced node features")
        
        # Show sample enhanced features
        if enhanced_features:
            sample_id = next(iter(enhanced_features))
            sample_feature = enhanced_features[sample_id]
            print("\\n📋 Sample Enhanced Node Features:")
            print(f"  - Label: {sample_feature.display_label}")
            print(f"  - Category: {sample_feature.category} > {sample_feature.subcategory}")
            print(f"  - Merchant: {sample_feature.merchant_clean}")
            print(f"  - Amount Bucket: {sample_feature.amount_bucket}")
            print(f"  - Time Bucket: {sample_feature.time_bucket}")
            print(f"  - Spending Pattern: {sample_feature.spending_pattern}")
            print(f"  - Frequency Score: {sample_feature.frequency_score:.3f}")
            print(f"  - Semantic Tags: {sample_feature.semantic_tags}")
        
        # Test enhanced graph building
        print("\\n🕸️  Building enhanced graph...")
        graph_service.build_enhanced_graph(transactions, db)
        
        # Test graph statistics
        print("\\n📈 Enhanced Graph Statistics:")
        stats = graph_service.get_graph_insights()
        
        print(f"  Basic Stats:")
        basic = stats.get('basic_stats', {})
        print(f"    - Nodes: {basic.get('nodes', 0)}")
        print(f"    - Edges: {basic.get('edges', 0)}")
        print(f"    - Density: {basic.get('density', 0):.4f}")
        print(f"    - Connected Components: {basic.get('connected_components', 0)}")
        
        # Pattern insights
        detected_patterns = stats.get('detected_patterns', [])
        if detected_patterns:
            print(f"\\n  💡 Detected Patterns:")
            for pattern in detected_patterns[:5]:  # Show top 5
                print(f"    - {pattern.get('pattern_name', 'Unknown')}: {pattern.get('description', 'N/A')}")
                print(f"      Confidence: {pattern.get('confidence', 0):.2f}, Transactions: {pattern.get('transaction_count', 0)}")
        
        # Category distribution
        categories = stats.get('category_distribution', {})
        if categories:
            print(f"\\n  �️  Category Distribution:")
            for category, count in list(categories.items())[:5]:  # Show top 5
                print(f"    - {category}: {count} transactions")
        
        # Spending patterns
        spending_patterns = stats.get('spending_patterns', {})
        if spending_patterns:
            print(f"\\n  🛒 Spending Patterns:")
            for pattern, count in spending_patterns.items():
                print(f"    - {pattern}: {count} transactions")
        
        # Temporal insights
        temporal = stats.get('temporal_insights', {})
        if temporal:
            print(f"\\n  ⏰ Temporal Insights:")
            print(f"    - Weekend transactions: {temporal.get('weekend_transactions', 0)}")
            print(f"    - Weekday transactions: {temporal.get('weekday_transactions', 0)}")
            print(f"    - Weekend ratio: {temporal.get('weekend_ratio', 0):.1%}")
        
        # Merchant insights
        merchant_insights = stats.get('merchant_insights', {})
        if merchant_insights:
            print(f"\\n  � Merchant Insights:")
            print(f"    - Unique merchants: {merchant_insights.get('unique_merchants', 0)}")
            
            top_frequency = merchant_insights.get('top_by_frequency', [])
            if top_frequency:
                print(f"    Top by frequency:")
                for merchant, count in top_frequency[:3]:
                    print(f"      - {merchant[:30]}...: {count} transactions")
            
            top_amount = merchant_insights.get('top_by_amount', [])
            if top_amount:
                print(f"    Top by amount:")
                for merchant, amount in top_amount[:3]:
                    print(f"      - {merchant[:30]}...: ${amount:.2f}")
        
        # Test pattern detection
        print("\\n🔍 Testing Pattern Detection...")
        patterns = graph_service.detected_patterns
        if patterns:
            print(f"✅ Detected {len(patterns)} spending patterns")
            for pattern in patterns[:3]:  # Show first 3
                print(f"  - {pattern.pattern_name}: {pattern.description}")
                print(f"    Confidence: {pattern.confidence:.2f}, Transactions: {len(pattern.transactions)}")
        else:
            print("⚠️  No specific patterns detected")
        
        # Test querying
        print("\\n🔎 Testing Pattern Queries...")
        
        # Create a simple query method for testing
        def simple_query(query_text):
            results = []
            query_lower = query_text.lower()
            
            if 'food' in query_lower:
                food_nodes = [node for node in graph_service.graph.nodes() 
                             if 'food' in graph_service.graph.nodes[node].get('category', '').lower() or
                                'restaurant' in graph_service.graph.nodes[node].get('category', '').lower()]
                results.append({'pattern': 'Food/Restaurant transactions', 'count': len(food_nodes)})
            
            if 'large' in query_lower:
                large_nodes = [node for node in graph_service.graph.nodes() 
                              if 'XLarge' in graph_service.graph.nodes[node].get('amount_bucket', '')]
                results.append({'pattern': 'Large purchases', 'count': len(large_nodes)})
            
            if 'weekend' in query_lower:
                weekend_nodes = [node for node in graph_service.graph.nodes() 
                                if 'Weekend' in graph_service.graph.nodes[node].get('time_bucket', '')]
                results.append({'pattern': 'Weekend transactions', 'count': len(weekend_nodes)})
            
            return results
        
        test_queries = [
            "food spending patterns",
            "large purchases", 
            "weekend transactions"
        ]
        
        for query in test_queries:
            print(f"\\n  Query: '{query}'")
            results = simple_query(query)
            if results:
                for result in results[:2]:  # Show top 2 results
                    print(f"    ✅ {result.get('pattern', 'Unknown')}: {result.get('count', 0)} items")
            else:
                print(f"    ⚠️  No results found")
        
        # Test visualization capabilities
        print("\\n🎨 Testing Enhanced Visualizations...")
        
        try:
            # Test different visualization types
            viz_types = ["spring", "community", "kamada_kawai"]
            color_schemes = ["category", "amount", "pattern"]
            
            for viz_type in viz_types:
                for color_scheme in color_schemes:
                    fig = visualizer.create_enhanced_visualization(
                        graph_service, 
                        layout_type=viz_type,
                        color_by=color_scheme,
                        size_by="amount"
                    )
                    
                    if fig and hasattr(fig, 'data') and fig.data:
                        print(f"    ✅ {viz_type.title()} layout with {color_scheme} coloring: {len(fig.data)} traces")
                    else:
                        print(f"    ⚠️  {viz_type.title()} layout with {color_scheme} coloring: Failed")
            
            # Test pattern summary
            summary_fig = visualizer.create_pattern_summary_chart(graph_service)
            if summary_fig and hasattr(summary_fig, 'data'):
                print(f"    ✅ Pattern summary chart: {len(summary_fig.data)} traces")
            
            # Test community visualization
            community_fig = visualizer.create_community_visualization(graph_service)
            if community_fig and hasattr(community_fig, 'data'):
                print(f"    ✅ Community visualization: {len(community_fig.data)} traces")
            
        except Exception as e:
            print(f"    ❌ Visualization error: {e}")
        
        # Test anomaly detection
        print("\\n🚨 Testing Anomaly Detection...")
        try:
            # Simple anomaly detection based on centrality and amount
            anomalies = []
            for node_id in graph_service.graph.nodes():
                node_data = graph_service.graph.nodes[node_id]
                centrality = node_data.get('centrality_score', 0)
                amount_bucket = node_data.get('amount_bucket', '')
                
                # High centrality + large amount = potential anomaly
                if centrality > 0.1 and 'XLarge' in amount_bucket:
                    anomalies.append({
                        'node_id': node_id,
                        'description': f"High influence large transaction: {node_data.get('merchant', 'Unknown')}",
                        'confidence': centrality
                    })
            
            if anomalies:
                print(f"✅ Detected {len(anomalies)} potential anomalies")
                for anomaly in anomalies[:3]:  # Show first 3
                    print(f"  - {anomaly.get('description', 'Unknown anomaly')}")
                    print(f"    Confidence: {anomaly.get('confidence', 0):.2f}")
            else:
                print("⚠️  No anomalies detected")
        except Exception as e:
            print(f"❌ Anomaly detection error: {e}")
        
        print("\\n✅ Enhanced Graph Testing Complete!")
        print("\\n📊 Summary:")
        print(f"  - Enhanced nodes: {len(enhanced_features)}")
        print(f"  - Graph nodes: {graph_service.graph.number_of_nodes()}")
        print(f"  - Graph edges: {graph_service.graph.number_of_edges()}")
        print(f"  - Detected patterns: {len(patterns) if patterns else 0}")
        print(f"  - Categories found: {len(stats.get('category_distribution', {}))}")
        print(f"  - Unique merchants: {stats.get('merchant_insights', {}).get('unique_merchants', 0)}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    test_enhanced_graph_functionality()
