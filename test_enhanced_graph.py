#!/usr/bin/env python3
"""
Test script for enhanced graph database features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_graph_imports():
    """Test if enhanced graph services can be imported."""
    print("Testing enhanced graph database imports...")
    
    try:
        from src.enhanced_graph_service import EnhancedGraphService
        print("✅ EnhancedGraphService imported successfully")
        
        from src.enhanced_graph_visualizer import EnhancedGraphVisualizer
        print("✅ EnhancedGraphVisualizer imported successfully")
        
        # Test initialization
        service = EnhancedGraphService()
        print("✅ EnhancedGraphService initialized successfully")
        
        visualizer = EnhancedGraphVisualizer()
        print("✅ EnhancedGraphVisualizer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_node_features():
    """Test enhanced node features creation."""
    print("\nTesting enhanced node features...")
    
    try:
        from src.enhanced_graph_service import EnhancedGraphService
        from src.models import Transaction
        from datetime import datetime
        
        service = EnhancedGraphService()
        
        # Create mock transactions
        transactions = [
            Transaction(
                id=1,
                date=datetime(2024, 1, 15, 14, 30),
                amount=-45.67,
                transaction_type="debit",
                merchant="WALMART SUPERCENTER",
                description="Grocery shopping"
            ),
            Transaction(
                id=2,
                date=datetime(2024, 1, 16, 19, 15),
                amount=-12.50,
                transaction_type="debit", 
                merchant="STARBUCKS COFFEE",
                description="Coffee"
            )
        ]
        
        # Add mock embedding
        transactions[0].embedding = [0.1] * 1536
        transactions[1].embedding = [0.2] * 1536
        
        # Test node features creation
        features = service.create_enhanced_node_features(transactions)
        print(f"✅ Created features for {len(features)} transactions")
        
        # Test feature details
        for trans_id, feature in features.items():
            print(f"Transaction {trans_id}:")
            print(f"  Label: {feature.display_label}")
            print(f"  Category: {feature.category}")
            print(f"  Pattern: {feature.spending_pattern}")
            print(f"  Tags: {feature.semantic_tags}")
        
        return True
        
    except Exception as e:
        print(f"❌ Node features test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_detection():
    """Test pattern detection functionality."""
    print("\nTesting pattern detection...")
    
    try:
        from src.enhanced_graph_service import EnhancedGraphService
        from src.models import Transaction
        from datetime import datetime, timedelta
        
        service = EnhancedGraphService()
        
        # Create mock transactions with patterns
        base_date = datetime(2024, 1, 1)
        transactions = []
        
        # Recurring payments (monthly Netflix)
        for i in range(6):
            trans = Transaction(
                id=i+1,
                date=base_date + timedelta(days=30*i),
                amount=-15.99,
                transaction_type="debit",
                merchant="NETFLIX.COM",
                description="Subscription"
            )
            transactions.append(trans)
        
        # Spending burst
        burst_date = base_date + timedelta(days=100)
        for i in range(4):
            trans = Transaction(
                id=i+7,
                date=burst_date + timedelta(days=i),
                amount=-200 - i*50,
                transaction_type="debit",
                merchant=f"STORE_{i}",
                description="Shopping"
            )
            transactions.append(trans)
        
        # Test pattern detection
        patterns = service.detect_transaction_patterns(transactions)
        print(f"✅ Detected {len(patterns)} patterns")
        
        for pattern in patterns:
            print(f"Pattern: {pattern.pattern_name}")
            print(f"  Type: {pattern.pattern_type}")
            print(f"  Confidence: {pattern.confidence:.3f}")
            print(f"  Transactions: {len(pattern.transactions)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Enhanced Graph Database Test Suite")
    print("=" * 50)
    
    tests = [
        test_enhanced_graph_imports,
        test_node_features,
        test_pattern_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced graph database is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
