import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import json

from src.models import get_db, Transaction, TransactionCluster
from src.fallback_query_interface import FallbackQueryInterface
from src.graph_service import GraphService
from src.embedding_service import EmbeddingService

# Page configuration
st.set_page_config(
    page_title="Graph-Powered Transaction Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def init_services():
    return FallbackQueryInterface(), GraphService(), EmbeddingService()

query_interface, graph_service, embedding_service = init_services()

# Title and description
st.title("💰 Graph-Powered Transaction Analytics")
st.markdown("""
**Turn your transaction data into actionable insights using AI and graph analytics.**

Ask questions in plain English and get intelligent answers backed by your financial data.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a view:",
    ["💬 Chat Interface", "📊 Dashboard", "🕸️ Graph Explorer", "🔍 Transaction Search", "📈 Analytics"]
)

# Get database session
db = next(get_db())

# Main content based on page selection
if page == "💬 Chat Interface":
    st.header("Ask Questions About Your Transactions")
    
    # Check if we're in mock mode
    if hasattr(query_interface, 'use_mock') and query_interface.use_mock:
        st.info("🔄 **Demo Mode**: Using keyword-based query processing. For full AI capabilities, add OpenAI credits to your account.")
    
    # Sample questions
    st.markdown("**Try asking questions like:**")
    sample_questions = [
        "What did I spend the most money on?",
        "Show me all transactions related to coffee or food",
        "What's my monthly spending summary?",
        "Find unusual or anomalous transactions",
        "What are my top 5 merchants by spending?",
        "Show me investment transactions",
        "What transaction patterns can you detect?"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.button(question, key=f"sample_{i}"):
            st.session_state.user_query = question
    
    # Chat interface
    user_query = st.text_input(
        "Ask a question about your transactions:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., How much did I spend on groceries last month?"
    )
    
    if st.button("Ask", type="primary") and user_query:
        with st.spinner("Analyzing your transactions..."):
            result = query_interface.process_natural_language_query(user_query, db)
        
        st.markdown("### Answer")
        st.write(result.get('response', 'No response generated'))
        
        if result.get('data'):
            st.markdown("### Detailed Data")
            
            # Display data based on type
            data = result['data']
            if isinstance(data, dict):
                if 'total_transactions' in data:
                    # Summary data
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", data.get('total_transactions', 0))
                    with col2:
                        st.metric("Total Income", f"${data.get('total_income', 0):,.2f}")
                    with col3:
                        st.metric("Total Expenses", f"${data.get('total_expenses', 0):,.2f}")
                    with col4:
                        st.metric("Net Amount", f"${data.get('net_amount', 0):,.2f}")
                
                else:
                    st.json(data)
            
            elif isinstance(data, list):
                if data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.json(data)
        
        if result.get('error'):
            st.error(f"Error: {result['error']}")

elif page == "📊 Dashboard":
    st.header("Financial Dashboard")
    
    # Load summary statistics
    transactions = db.query(Transaction).all()
    
    if not transactions:
        st.warning("No transactions found. Please run the data loading script first.")
    else:
        # Summary metrics
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = sum(t.amount for t in transactions if t.amount < 0)
        net_amount = total_income + total_expenses
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(transactions))
        with col2:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col3:
            st.metric("Total Expenses", f"${abs(total_expenses):,.2f}")
        with col4:
            st.metric("Net Amount", f"${net_amount:,.2f}", delta=f"{net_amount:,.2f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by category
            st.subheader("Spending by Category")
            category_data = {}
            for t in transactions:
                if t.transaction_type and t.amount < 0:
                    category = t.transaction_type
                    category_data[category] = category_data.get(category, 0) + abs(t.amount)
            
            if category_data:
                fig = px.pie(
                    values=list(category_data.values()),
                    names=list(category_data.keys()),
                    title="Expense Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly trend
            st.subheader("Monthly Spending Trend")
            df = pd.DataFrame([
                {
                    'date': t.date,
                    'amount': t.amount,
                    'type': 'Income' if t.amount > 0 else 'Expense'
                }
                for t in transactions
            ])
            
            if not df.empty:
                df['month'] = df['date'].dt.to_period('M')
                monthly = df.groupby(['month', 'type'])['amount'].sum().reset_index()
                monthly['month_str'] = monthly['month'].astype(str)
                
                fig = px.bar(
                    monthly,
                    x='month_str',
                    y='amount',
                    color='type',
                    title="Monthly Income vs Expenses"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Top merchants
        st.subheader("Top Merchants by Spending")
        merchant_data = {}
        for t in transactions:
            if t.merchant and t.amount < 0:
                merchant = t.merchant.strip()
                merchant_data[merchant] = merchant_data.get(merchant, 0) + abs(t.amount)
        
        if merchant_data:
            top_merchants = sorted(merchant_data.items(), key=lambda x: x[1], reverse=True)[:10]
            df_merchants = pd.DataFrame(top_merchants, columns=['Merchant', 'Total Spent'])
            st.dataframe(df_merchants, use_container_width=True)

elif page == "🕸️ Graph Explorer":
    st.header("Transaction Graph Explorer")
    
    transactions = db.query(Transaction).all()
    
    if not transactions:
        st.warning("No transactions found. Please run the data loading script first.")
    else:
        # Build/load graph
        if not graph_service.graph.nodes():
            with st.spinner("Building transaction graph..."):
                graph_service.build_graph_from_transactions(transactions, db)
        
        # Graph statistics
        stats = graph_service.get_graph_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", stats.get('nodes', 0))
        with col2:
            st.metric("Edges", stats.get('edges', 0))
        with col3:
            st.metric("Density", f"{stats.get('density', 0):.3f}")
        
        # Graph visualization (simplified)
        st.subheader("Graph Visualization")
        
        if stats.get('nodes', 0) > 0:
            # Sample a subset for visualization if too large
            G = graph_service.graph
            if G.number_of_nodes() > 100:
                # Sample nodes
                sample_nodes = list(G.nodes())[:100]
                G_sample = G.subgraph(sample_nodes)
            else:
                G_sample = G
            
            # Create layout
            pos = nx.spring_layout(G_sample, k=1, iterations=50)
            
            # Prepare data for plotly
            edge_x = []
            edge_y = []
            for edge in G_sample.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in G_sample.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get transaction data
                transaction = next((t for t in transactions if t.id == node), None)
                if transaction:
                    node_text.append(f"ID: {transaction.id}<br>Amount: ${transaction.amount}<br>Type: {transaction.transaction_type}")
                    node_color.append(transaction.amount)
                else:
                    node_text.append(f"ID: {node}")
                    node_color.append(0)
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='lightgray'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=8,
                    color=node_color,
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Amount")
                )
            ))
            
            fig.update_layout(
                title="Transaction Graph Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Nodes represent transactions, edges show relationships",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Clusters
        st.subheader("Transaction Clusters")
        clusters = db.query(TransactionCluster).all()
        
        if clusters:
            for cluster in clusters:
                with st.expander(f"{cluster.name} ({len(cluster.transaction_ids)} transactions)"):
                    st.write(cluster.description)
                    if cluster.transaction_ids:
                        sample_transactions = db.query(Transaction).filter(
                            Transaction.id.in_(cluster.transaction_ids[:5])
                        ).all()
                        
                        for t in sample_transactions:
                            st.write(f"• ${t.amount} - {t.transaction_type} - {t.merchant}")

elif page == "🔍 Transaction Search":
    st.header("Semantic Transaction Search")
    
    search_query = st.text_input(
        "Search transactions by description:",
        placeholder="e.g., coffee, grocery, gas, investment"
    )
    
    if search_query:
        with st.spinner("Searching transactions..."):
            similar_transactions = embedding_service.semantic_search(search_query, db, limit=20)
        
        if similar_transactions:
            st.subheader(f"Found {len(similar_transactions)} similar transactions")
            
            # Display results
            results_data = []
            for t in similar_transactions:
                results_data.append({
                    'Date': t.date.strftime('%Y-%m-%d'),
                    'Amount': f"${t.amount:.2f}",
                    'Type': t.transaction_type,
                    'Merchant': t.merchant,
                    'Description': t.description
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.info("No similar transactions found.")

elif page == "📈 Analytics":
    st.header("Advanced Analytics")
    
    transactions = db.query(Transaction).all()
    
    if not transactions:
        st.warning("No transactions found. Please run the data loading script first.")
    else:
        # Anomaly detection
        st.subheader("Anomaly Detection")
        
        if st.button("Detect Anomalous Transactions"):
            with st.spinner("Analyzing transaction patterns..."):
                anomalies = query_interface.detect_anomalies(db, threshold=90)
            
            if anomalies:
                st.warning(f"Found {len(anomalies)} potentially anomalous transactions:")
                
                anomaly_data = []
                for anomaly in anomalies:
                    anomaly_data.append({
                        'Date': anomaly['date'],
                        'Amount': f"${anomaly['amount']:.2f}",
                        'Type': anomaly['transaction_type'],
                        'Merchant': anomaly['merchant'],
                        'Reason': anomaly['reason']
                    })
                
                df_anomalies = pd.DataFrame(anomaly_data)
                st.dataframe(df_anomalies, use_container_width=True)
            else:
                st.success("No anomalous transactions detected!")
        
        # Time series analysis
        st.subheader("Spending Patterns Over Time")
        
        df = pd.DataFrame([
            {
                'date': t.date,
                'amount': abs(t.amount) if t.amount < 0 else 0,  # Only expenses
                'category': t.transaction_type
            }
            for t in transactions
        ])
        
        if not df.empty:
            # Daily spending
            daily_spending = df.groupby('date')['amount'].sum().reset_index()
            
            fig = px.line(
                daily_spending,
                x='date',
                y='amount',
                title="Daily Spending Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category trends
            if st.checkbox("Show category trends"):
                category_trends = df.groupby(['date', 'category'])['amount'].sum().reset_index()
                
                fig = px.line(
                    category_trends,
                    x='date',
                    y='amount',
                    color='category',
                    title="Spending Trends by Category"
                )
                st.plotly_chart(fig, use_container_width=True)

# Close database connection
db.close()

# Footer
st.markdown("---")
st.markdown("**Graph-Powered Transaction Analytics** - Built with Streamlit, OpenAI, and PyTorch Geometric")
