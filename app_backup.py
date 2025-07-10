import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import json

from src.models import get_db, Transaction, TransactionCluster
from src.smart_query_interface import SmartQueryInterface
from src.graph_service import GraphService
from src.embedding_service import EmbeddingService
from src.thought_journal_service import ThoughtJournalService

# Page configuration
st.set_page_config(
    page_title="Second Brain",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def init_services():
    return SmartQueryInterface(), GraphService(), EmbeddingService(), ThoughtJournalService()

query_interface, graph_service, embedding_service, journal_service = init_services()

# Title and description
st.title("Second Brain")
st.markdown("""
**Turn your transaction data into actionable insights using AI and graph analytics.**

Ask questions in plain English and get intelligent answers backed by your financial data.
""")

# Sidebar
st.sidebar.title("Navigation")
# Use sidebar buttons for navigation instead of dropdown or radio
pages = [
    "💬 Chat Interface",
    "📓 Thought Journal",
    "📊 Dashboard",
    "🕸️ Graph Explorer",
    "🔍 Transaction Search",
    "📈 Analytics"
]

if "page" not in st.session_state:
    st.session_state.page = pages[0]

for p in pages:
    if st.sidebar.button(p, key=f"nav_{p}"):
        st.session_state.page = p

page = st.session_state.page

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
            result = query_interface.process_query(user_query, db)
        
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

elif page == "📓 Thought Journal":
    st.header("💭 Thought Journal")
    st.markdown("Record your thoughts, voice notes, and images with automatic linking to your transactions.")
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["✍️ New Entry", "📖 View Entries", "🔗 Linked Transactions", "📊 Analytics"])
    
    with tab1:
        st.subheader("Create New Journal Entry")
        
        # Entry type selection
        entry_type = st.selectbox(
            "Entry Type:",
            ["Text", "Audio", "Image", "Mixed"],
            help="Choose the type of journal entry you want to create"
        )
        
        # Common fields
        title = st.text_input("Title", placeholder="Enter a title for your entry...")
        
        col1, col2 = st.columns(2)
        with col1:
            mood_score = st.slider("Mood", -1.0, 1.0, 0.0, 0.1, 
                                  help="How are you feeling? (-1 = negative, 0 = neutral, 1 = positive)")
        with col2:
            importance_score = st.slider("Importance", 0.0, 1.0, 0.5, 0.1,
                                       help="How important is this entry? (0 = low, 1 = high)")
        
        tags = st.text_input("Tags (comma-separated)", placeholder="work, personal, shopping, etc.")
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Entry-specific fields
        content = None
        audio_data = None
        audio_filename = None
        audio_transcript = None
        image_data = None
        image_filename = None
        image_description = None
        
        if entry_type in ["Text", "Mixed"]:
            content = st.text_area("Content", placeholder="Write your thoughts here...", height=200)
        
        if entry_type in ["Audio", "Mixed"]:
            st.markdown("**Audio Recording:**")
            audio_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'ogg', 'm4a'])
            if audio_file:
                audio_data = audio_file.read()
                audio_filename = audio_file.name
                st.audio(audio_data, format='audio/wav')
                
                # Audio transcript
                audio_transcript = st.text_area("Audio Transcript", 
                                               placeholder="Transcribe your audio here (optional)...",
                                               help="You can manually transcribe your audio or leave empty")
        
        if entry_type in ["Image", "Mixed"]:
            st.markdown("**Image:**")
            image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'gif'])
            if image_file:
                image_data = image_file.read()
                image_filename = image_file.name
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                
                # Image description
                image_description = st.text_area("Image Description", 
                                                placeholder="Describe what's in the image...",
                                                help="Describe the image content for better search and linking")
        
        # Create entry button
        if st.button("💾 Create Journal Entry", type="primary"):
            if not title:
                st.error("Please enter a title for your entry.")
            else:
                try:
                    with st.spinner("Creating journal entry..."):
                        if entry_type == "Text":
                            if not content:
                                st.error("Please enter some content for your text entry.")
                            else:
                                entry = journal_service.create_text_entry(
                                    db, title, content, tag_list, mood_score, importance_score
                                )
                        elif entry_type == "Audio":
                            if not audio_data:
                                st.error("Please upload an audio file.")
                            else:
                                entry = journal_service.create_audio_entry(
                                    db, title, audio_data, audio_filename, audio_transcript,
                                    tag_list, mood_score, importance_score
                                )
                        elif entry_type == "Image":
                            if not image_data:
                                st.error("Please upload an image file.")
                            else:
                                entry = journal_service.create_image_entry(
                                    db, title, image_data, image_filename, image_description,
                                    tag_list, mood_score, importance_score
                                )
                        elif entry_type == "Mixed":
                            if not any([content, audio_data, image_data]):
                                st.error("Please provide at least one type of content (text, audio, or image).")
                            else:
                                entry = journal_service.create_mixed_entry(
                                    db, title, content, audio_data, audio_filename, 
                                    audio_transcript, image_data, image_filename, 
                                    image_description, tag_list, mood_score, importance_score
                                )
                    
                    st.success(f"✅ Journal entry '{title}' created successfully!")
                    
                    # Show auto-linked transactions
                    if entry.auto_linked_transaction_ids:
                        st.info(f"🔗 Automatically linked to {len(entry.auto_linked_transaction_ids)} similar transactions")
                
                except Exception as e:
                    st.error(f"Error creating journal entry: {str(e)}")
    
    with tab2:
        st.subheader("Your Journal Entries")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Type", ["All", "text", "audio", "image", "mixed"])
        with col2:
            filter_tags = st.text_input("Filter by Tags", placeholder="Enter tags to filter...")
        with col3:
            limit = st.number_input("Number of entries", min_value=1, max_value=100, value=20)
        
        # Get entries
        filter_type_val = None if filter_type == "All" else filter_type
        filter_tags_list = [tag.strip() for tag in filter_tags.split(",") if tag.strip()] if filter_tags else None
        
        entries = journal_service.get_journal_entries(
            db, limit=limit, entry_type=filter_type_val, tags=filter_tags_list
        )
        
        if entries:
            st.write(f"Found {len(entries)} entries:")
            
            for entry in entries:
                with st.expander(f"📝 {entry.title} ({entry.entry_type}) - {entry.created_at.strftime('%Y-%m-%d %H:%M')}"):
                    
                    # Entry details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Type:** {entry.entry_type.title()}")
                        if entry.mood_score is not None:
                            mood_emoji = "😊" if entry.mood_score > 0.3 else "😐" if entry.mood_score > -0.3 else "😔"
                            st.write(f"**Mood:** {mood_emoji} {entry.mood_score:.1f}")
                    with col2:
                        if entry.importance_score is not None:
                            importance_stars = "⭐" * int(entry.importance_score * 5)
                            st.write(f"**Importance:** {importance_stars} {entry.importance_score:.1f}")
                    with col3:
                        if entry.tags:
                            st.write(f"**Tags:** {', '.join(entry.tags)}")
                    
                    # Content
                    if entry.content:
                        st.markdown(f"**Content:** {entry.content}")
                    
                    # Audio
                    if entry.audio_data:
                        st.markdown("**Audio:**")
                        st.audio(entry.audio_data, format='audio/wav')
                        if entry.audio_transcript:
                            st.markdown(f"**Transcript:** {entry.audio_transcript}")
                    
                    # Image
                    if entry.image_data:
                        st.markdown("**Image:**")
                        st.image(entry.image_data, caption=entry.image_description or "Journal Image", width=300)
                        if entry.image_description:
                            st.markdown(f"**Description:** {entry.image_description}")
                    
                    # Linked transactions
                    if entry.auto_linked_transaction_ids:
                        st.info(f"🔗 Linked to {len(entry.auto_linked_transaction_ids)} transactions")
                    
                    # Action buttons
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"✏️ Edit", key=f"edit_{entry.id}"):
                            st.session_state[f"edit_entry_{entry.id}"] = True
                    with col_delete:
                        if st.button(f"🗑️ Delete", key=f"delete_{entry.id}"):
                            if st.session_state.get(f"confirm_delete_{entry.id}", False):
                                journal_service.delete_journal_entry(db, entry.id)
                                st.success("Entry deleted!")
                                st.rerun()
                            else:
                                st.session_state[f"confirm_delete_{entry.id}"] = True
                                st.warning("Click again to confirm deletion")
        else:
            st.info("No journal entries found. Create your first entry in the 'New Entry' tab!")
    
    with tab3:
        st.subheader("Transaction Links")
        
        # Search for journal entries
        search_query = st.text_input("Search journal entries:", placeholder="Search by content...")
        
        if search_query:
            search_results = journal_service.search_journal_entries(db, search_query, limit=10)
            
            if search_results:
                st.write(f"Found {len(search_results)} matching entries:")
                
                for entry, similarity in search_results:
                    with st.expander(f"📝 {entry.title} (Similarity: {similarity:.3f})"):
                        st.write(f"**Type:** {entry.entry_type}")
                        st.write(f"**Content:** {entry.content[:200]}..." if entry.content else "No text content")
                        st.write(f"**Created:** {entry.created_at.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Show linked transactions
                        linked_transactions = journal_service.get_linked_transactions(db, entry.id)
                        
                        if linked_transactions:
                            st.write(f"**Linked Transactions ({len(linked_transactions)}):**")
                            
                            for link_data in linked_transactions:
                                transaction = link_data['transaction']
                                link = link_data['link']
                                
                                st.write(f"- {transaction.date.strftime('%Y-%m-%d')}: ${transaction.amount:.2f} - {transaction.merchant or 'Unknown'}")
                                st.write(f"  Similarity: {link.similarity_score:.3f} ({link.link_type})")
                        else:
                            st.write("No linked transactions")
        
        # Manual linking section
        st.markdown("---")
        st.subheader("Manual Transaction Linking")
        
        col1, col2 = st.columns(2)
        with col1:
            # Select journal entry
            all_entries = journal_service.get_journal_entries(db, limit=100)
            if all_entries:
                selected_entry = st.selectbox(
                    "Select Journal Entry:",
                    options=all_entries,
                    format_func=lambda x: f"{x.title} ({x.entry_type})"
                )
        
        with col2:
            # Select transaction
            recent_transactions = db.query(Transaction).order_by(Transaction.date.desc()).limit(100).all()
            if recent_transactions:
                selected_transaction = st.selectbox(
                    "Select Transaction:",
                    options=recent_transactions,
                    format_func=lambda x: f"{x.date.strftime('%Y-%m-%d')}: ${x.amount:.2f} - {x.merchant or 'Unknown'}"
                )
        
        link_reason = st.text_input("Link Reason (optional):", placeholder="Why are these related?")
        
        if st.button("🔗 Link Entry to Transaction"):
            if selected_entry and selected_transaction:
                try:
                    journal_service.manually_link_to_transaction(
                        db, selected_entry.id, selected_transaction.id, link_reason
                    )
                    st.success("Successfully linked journal entry to transaction!")
                except Exception as e:
                    st.error(f"Error linking: {str(e)}")
    
    with tab4:
        st.subheader("Journal Analytics")
        
        # Get analytics
        analytics = journal_service.get_journal_analytics(db)
        
        if analytics['total_entries'] > 0:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Entries", analytics['total_entries'])
            with col2:
                st.metric("Transaction Links", analytics['total_transaction_links'])
            with col3:
                if analytics['average_mood'] is not None:
                    mood_emoji = "😊" if analytics['average_mood'] > 0.3 else "😐" if analytics['average_mood'] > -0.3 else "😔"
                    st.metric("Average Mood", f"{mood_emoji} {analytics['average_mood']:.2f}")
                else:
                    st.metric("Average Mood", "N/A")
            with col4:
                st.metric("Unique Tags", len(analytics['top_tags']))
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Entry types chart
                if analytics['entries_by_type']:
                    fig = px.pie(
                        values=list(analytics['entries_by_type'].values()),
                        names=list(analytics['entries_by_type'].keys()),
                        title="Journal Entries by Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top tags chart
                if analytics['top_tags']:
                    tag_names = [tag for tag, count in analytics['top_tags'][:10]]
                    tag_counts = [count for tag, count in analytics['top_tags'][:10]]
                    
                    fig = px.bar(
                        x=tag_counts,
                        y=tag_names,
                        orientation='h',
                        title="Top Tags",
                        labels={'x': 'Count', 'y': 'Tags'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No journal entries yet. Create your first entry to see analytics!")

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
