import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import json
import base64
import io

from src.models import get_db, Transaction, TransactionCluster
from src.smart_query_interface import SmartQueryInterface
from src.graph_service import GraphService
from src.embedding_service import EmbeddingService
from src.enhanced_thought_journal_service import EnhancedThoughtJournalService
from src.chat_history_service import ChatHistoryService

# Page configuration
st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Mobile-friendly: collapsed sidebar
)

# Initialize services
@st.cache_resource
def init_services():
    return (SmartQueryInterface(), GraphService(), EmbeddingService(), 
            EnhancedThoughtJournalService(), ChatHistoryService())

query_interface, graph_service, embedding_service, journal_service, chat_service = init_services()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = False

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    .main-nav-button {
        width: 100%;
        height: 80px;
        margin: 10px 0;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .main-nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .thought-entry-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .big-button {
        width: 100%;
        height: 60px;
        font-size: 16px;
        margin: 10px 0;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .primary-button {
        background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    
    .secondary-button {
        background: linear-gradient(45deg, #6c757d 0%, #495057 100%);
        color: white;
    }
    
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .chat-message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: white;
        color: #333;
        border: 1px solid #dee2e6;
    }
    
    @media (max-width: 768px) {
        .main-nav-button {
            height: 70px;
            font-size: 16px;
        }
        
        .big-button {
            height: 50px;
            font-size: 14px;
        }
        
        .chat-container {
            height: 400px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Navigation function
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# Main navigation
page = st.session_state.page

# Get database session
db = next(get_db())

# Home/Landing Page
if page == "🏠 Home":
    st.title("🧠 Second Brain")
    st.markdown("""
    ### Your AI-Powered Financial Intelligence Platform
    
    Turn your transaction data into actionable insights using AI and graph analytics.
    Ask questions in plain English and get intelligent answers backed by your financial data.
    """)
    
    # Large navigation buttons for mobile
    st.markdown("### Choose Your Adventure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Chat with AI", key="nav_chat", help="Ask questions about your finances"):
            navigate_to("💬 Chat Interface")
            
        if st.button("📓 Thought Journal", key="nav_journal", help="Record thoughts and insights"):
            navigate_to("📓 Thought Journal")
            
        if st.button("📊 Dashboard", key="nav_dashboard", help="View financial overview"):
            navigate_to("📊 Dashboard")
    
    with col2:
        if st.button("🕸️ Graph Explorer", key="nav_graph", help="Explore transaction networks"):
            navigate_to("🕸️ Graph Explorer")
            
        if st.button("🔍 Search Transactions", key="nav_search", help="Find specific transactions"):
            navigate_to("🔍 Transaction Search")
            
        if st.button("📈 Analytics", key="nav_analytics", help="Advanced financial analysis"):
            navigate_to("📈 Analytics")
    
    # Quick stats
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    
    transactions = db.query(Transaction).all()
    if transactions:
        total_transactions = len(transactions)
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = sum(t.amount for t in transactions if t.amount < 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", total_transactions)
        with col2:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col3:
            st.metric("Total Expenses", f"${abs(total_expenses):,.2f}")
    else:
        st.info("No transactions found. Upload your data to get started!")

# Chat Interface with History
elif page == "💬 Chat Interface":
    st.title("💬 Chat with Your Financial AI")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_chat"):
        navigate_to("🏠 Home")
    
    # Chat layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Current Conversation")
        
        # Get or create current thread
        if not st.session_state.current_thread_id:
            thread = chat_service.create_thread(db, "New Chat")
            st.session_state.current_thread_id = thread.id
        
        # Display chat messages
        messages = chat_service.get_messages(db, st.session_state.current_thread_id)
        
        # Chat container
        chat_container = st.container()
        with chat_container:
            for msg in messages:
                if msg.message_type == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {msg.content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>AI:</strong> {msg.content}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Sample questions
        st.markdown("**💡 Try asking:**")
        sample_questions = [
            "What did I spend the most money on?",
            "Show me all transactions related to coffee or food",
            "What's my monthly spending summary?",
            "Find unusual or anomalous transactions",
            "What are my top 5 merchants by spending?"
        ]
        
        cols = st.columns(len(sample_questions))
        for i, question in enumerate(sample_questions):
            with cols[i % len(cols)]:
                if st.button(question, key=f"sample_{i}"):
                    st.session_state.user_query = question
        
        # Chat input
        user_query = st.text_input(
            "Ask a question about your transactions:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., How much did I spend on groceries last month?",
            key="chat_input"
        )
        
        if st.button("Send", type="primary", key="send_chat"):
            if user_query:
                with st.spinner("Thinking..."):
                    # Add user message
                    chat_service.add_message(db, st.session_state.current_thread_id, "user", user_query)
                    
                    # Get AI response
                    result = query_interface.process_query(user_query, db)
                    response = result.get('response', 'No response generated')
                    
                    # Add AI response
                    chat_service.add_message(db, st.session_state.current_thread_id, "assistant", response)
                    
                    # Clear input
                    st.session_state.user_query = ""
                    st.rerun()
    
    with col2:
        st.subheader("Chat History")
        
        # New chat button
        if st.button("➕ New Chat", key="new_chat"):
            thread = chat_service.create_thread(db, f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.session_state.current_thread_id = thread.id
            st.rerun()
        
        # List recent threads
        recent_threads = chat_service.get_recent_threads(db, limit=10)
        
        for thread in recent_threads:
            is_current = thread.id == st.session_state.current_thread_id
            if st.button(
                f"{'🔘' if is_current else '⚪'} {thread.thread_title}",
                key=f"thread_{thread.id}",
                disabled=is_current
            ):
                st.session_state.current_thread_id = thread.id
                st.rerun()

# Simplified Thought Journal
elif page == "📓 Thought Journal":
    st.title("📓 Thought Journal")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_journal"):
        navigate_to("🏠 Home")
    
    st.markdown("Record your thoughts, insights, and observations. The AI will automatically extract metadata and link to relevant transactions.")
    
    # Simplified entry form
    st.markdown("### ✍️ New Entry")
    
    with st.container():
        st.markdown('<div class="thought-entry-container">', unsafe_allow_html=True)
        
        # Single text box for all content
        entry_content = st.text_area(
            "What's on your mind?",
            placeholder="Share your thoughts, observations, or insights here...",
            height=150,
            key="thought_content"
        )
        
        # Big buttons row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Audio recording/upload
            st.markdown("🎤 **Audio**")
            audio_file = st.file_uploader("Record or Upload Audio", type=['mp3', 'wav', 'ogg', 'm4a'], key="audio_upload")
            if audio_file:
                st.audio(audio_file, format='audio/wav')
        
        with col2:
            # Image capture/upload
            st.markdown("📸 **Image**")
            image_file = st.file_uploader("Capture or Upload Image", type=['png', 'jpg', 'jpeg', 'gif'], key="image_upload")
            if image_file:
                st.image(image_file, caption="Uploaded Image", width=200)
        
        with col3:
            # Voice-to-text (placeholder for future feature)
            st.markdown("🗣️ **Voice-to-Text**")
            st.info("Coming soon: Record voice and convert to text automatically")
        
        # Create entry button
        if st.button("💾 Save Thought", type="primary", key="save_thought"):
            if entry_content or audio_file or image_file:
                try:
                    with st.spinner("Processing your thought..."):
                        # Prepare data for mixed entry
                        audio_data = audio_file.read() if audio_file else None
                        audio_filename = audio_file.name if audio_file else None
                        image_data = image_file.read() if image_file else None
                        image_filename = image_file.name if image_file else None
                        
                        # Create mixed entry - let the service handle metadata extraction
                        entry = journal_service.create_mixed_entry(
                            db=db,
                            title=f"Thought - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            content=entry_content or None,
                            audio_data=audio_data,
                            audio_filename=audio_filename,
                            audio_transcript=None,  # Will be extracted by service
                            image_data=image_data,
                            image_filename=image_filename,
                            image_description=None,  # Will be extracted by service
                            tags=None,  # Will be extracted by service
                            mood_score=None,  # Will be extracted by service
                            importance_score=None  # Will be extracted by service
                        )
                        
                        st.success("✅ Thought saved successfully!")
                        
                        # Show auto-extracted metadata
                        if entry.tags:
                            st.info(f"🏷️ Auto-extracted tags: {', '.join(entry.tags)}")
                        if entry.mood_score is not None:
                            mood_emoji = "😊" if entry.mood_score > 0.3 else "😐" if entry.mood_score > -0.3 else "😔"
                            st.info(f"😊 Detected mood: {mood_emoji} {entry.mood_score:.2f}")
                        if entry.auto_linked_transaction_ids:
                            st.info(f"🔗 Auto-linked to {len(entry.auto_linked_transaction_ids)} transactions")
                        
                        # Clear form
                        st.session_state.thought_content = ""
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error saving thought: {str(e)}")
            else:
                st.warning("Please enter some content, audio, or image before saving.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent entries
    st.markdown("### 📖 Recent Thoughts")
    
    recent_entries = journal_service.get_journal_entries(db, limit=5)
    
    if recent_entries:
        for entry in recent_entries:
            with st.expander(f"💭 {entry.title} - {entry.created_at.strftime('%Y-%m-%d %H:%M')}"):
                if entry.content:
                    st.markdown(f"**Content:** {entry.content}")
                
                if entry.audio_data:
                    st.markdown("**Audio:**")
                    st.audio(entry.audio_data, format='audio/wav')
                    if entry.audio_transcript:
                        st.markdown(f"**Transcript:** {entry.audio_transcript}")
                
                if entry.image_data:
                    st.markdown("**Image:**")
                    st.image(entry.image_data, caption=entry.image_description or "Journal Image", width=300)
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    if entry.tags:
                        st.markdown(f"**Tags:** {', '.join(entry.tags)}")
                with col2:
                    if entry.mood_score is not None:
                        mood_emoji = "😊" if entry.mood_score > 0.3 else "😐" if entry.mood_score > -0.3 else "😔"
                        st.markdown(f"**Mood:** {mood_emoji} {entry.mood_score:.2f}")
                with col3:
                    if entry.auto_linked_transaction_ids:
                        st.markdown(f"**Links:** {len(entry.auto_linked_transaction_ids)} transactions")
    else:
        st.info("No thoughts recorded yet. Start by sharing what's on your mind!")

# Keep existing pages with mobile improvements
elif page == "📊 Dashboard":
    st.title("📊 Financial Dashboard")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_dashboard"):
        navigate_to("🏠 Home")
    
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
    st.title("🕸️ Transaction Graph Explorer")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_graph"):
        navigate_to("🏠 Home")
    
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
        
        # Graph visualization (simplified for mobile)
        st.subheader("Graph Visualization")
        
        if stats.get('nodes', 0) > 0:
            # Sample a subset for visualization if too large
            G = graph_service.graph
            if G.number_of_nodes() > 50:  # Reduced for mobile
                sample_nodes = list(G.nodes())[:50]
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
                height=400,  # Smaller height for mobile
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Transaction Search":
    st.title("🔍 Semantic Transaction Search")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_search"):
        navigate_to("🏠 Home")
    
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
    st.title("📈 Advanced Analytics")
    
    # Back to home button
    if st.button("🏠 Home", key="back_home_analytics"):
        navigate_to("🏠 Home")
    
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
            fig.update_layout(height=400)  # Smaller height for mobile
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
                fig.update_layout(height=400)  # Smaller height for mobile
                st.plotly_chart(fig, use_container_width=True)

# Close database connection
db.close()

# Footer
st.markdown("---")
st.markdown("**🧠 Second Brain** - Your AI-Powered Financial Intelligence Platform")
