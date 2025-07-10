import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from .config import Config
from .models import Transaction, GraphEdge, TransactionCluster
from .embedding_service import EmbeddingService

class AdvancedTransactionGNN(nn.Module):
    """Advanced Graph Neural Network for transaction analysis with multiple tasks."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(AdvancedTransactionGNN, self).__init__()
        
        # Graph attention layers for better representation
        self.gat1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * 8, hidden_dim, heads=4, dropout=0.1)
        self.gat3 = GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.1)
        
        # Task-specific heads
        self.anomaly_head = nn.Linear(output_dim, 1)  # Anomaly detection
        self.cluster_head = nn.Linear(output_dim, 10)  # Cluster prediction
        self.category_head = nn.Linear(output_dim, 20)  # Category prediction
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None, task='embedding'):
        # Graph attention layers
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = self.gat3(x, edge_index)
        
        if task == 'embedding':
            return x
        elif task == 'anomaly':
            return torch.sigmoid(self.anomaly_head(x))
        elif task == 'cluster':
            return F.softmax(self.cluster_head(x), dim=-1)
        elif task == 'category':
            return F.softmax(self.category_head(x), dim=-1)
        else:
            return x

class InteractiveGraphService:
    """Enhanced graph service with interactive visualizations and advanced GNN."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.graph = nx.Graph()
        self.gnn_model = None
        self.node_embeddings = {}
        self.graph_data = None
        
    def build_graph_from_transactions(self, transactions: List[Transaction], db: Session):
        """Build comprehensive graph with multiple edge types."""
        print("Building advanced graph from transactions...")
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes with rich features
        for transaction in transactions:
            # Create feature vector from transaction
            features = self._create_node_features(transaction)
            
            self.graph.add_node(
                transaction.id,
                transaction=transaction,
                amount=transaction.amount,
                date=transaction.date,
                type=transaction.transaction_type,
                merchant=transaction.merchant,
                features=features,
                embedding=transaction.embedding
            )
        
        # Add multiple types of edges
        self._add_similarity_edges(transactions, db)
        self._add_temporal_edges(transactions, db)
        self._add_merchant_edges(transactions, db)
        self._add_amount_edges(transactions, db)
        
        print(f"Advanced graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Prepare data for GNN
        self._prepare_graph_data(transactions)
        
    def _create_node_features(self, transaction: Transaction) -> List[float]:
        """Create feature vector for a transaction node."""
        features = []
        
        # Amount features
        features.append(transaction.amount)
        features.append(abs(transaction.amount))
        features.append(1.0 if transaction.amount > 0 else 0.0)  # is_income
        
        # Date features
        features.append(transaction.date.weekday())  # day of week
        features.append(transaction.date.day)  # day of month
        features.append(transaction.date.month)  # month
        
        # Category features (one-hot encoded)
        categories = ['Payroll', 'POS', 'Bill', 'Transfer', 'Investment', 'Loan', 'Insurance', 'Service']
        for cat in categories:
            features.append(1.0 if transaction.transaction_type and cat.lower() in transaction.transaction_type.lower() else 0.0)
        
        # Add embedding if available
        if transaction.embedding:
            features.extend(transaction.embedding)
        else:
            features.extend([0.0] * Config.EMBEDDING_DIMENSION)
            
        return features
    
    def _add_similarity_edges(self, transactions: List[Transaction], db: Session):
        """Add edges based on embedding similarity."""
        print("Adding similarity edges...")
        
        for i, trans1 in enumerate(transactions):
            if not trans1.embedding:
                continue
                
            for j, trans2 in enumerate(transactions[i+1:], i+1):
                if not trans2.embedding:
                    continue
                
                similarity = self.embedding_service.cosine_similarity(
                    trans1.embedding, trans2.embedding
                )
                
                if similarity >= Config.SIMILARITY_THRESHOLD:
                    self.graph.add_edge(
                        trans1.id, trans2.id,
                        weight=similarity,
                        edge_type='similarity'
                    )
    
    def _add_temporal_edges(self, transactions: List[Transaction], db: Session):
        """Add edges based on temporal proximity."""
        print("Adding temporal edges...")
        
        # Sort transactions by date
        sorted_transactions = sorted(transactions, key=lambda x: x.date)
        
        for i, trans1 in enumerate(sorted_transactions):
            for j in range(i+1, min(i+6, len(sorted_transactions))):  # Connect to next 5 transactions
                trans2 = sorted_transactions[j]
                
                # Calculate time difference in days
                time_diff = (trans2.date - trans1.date).days
                
                if time_diff <= 7:  # Within a week
                    weight = 1.0 / (time_diff + 1)  # Closer in time = higher weight
                    
                    self.graph.add_edge(
                        trans1.id, trans2.id,
                        weight=weight,
                        edge_type='temporal'
                    )
    
    def _add_merchant_edges(self, transactions: List[Transaction], db: Session):
        """Add edges between transactions from same merchants."""
        print("Adding merchant edges...")
        
        merchant_groups = {}
        for transaction in transactions:
            if transaction.merchant:
                if transaction.merchant not in merchant_groups:
                    merchant_groups[transaction.merchant] = []
                merchant_groups[transaction.merchant].append(transaction)
        
        for merchant, group in merchant_groups.items():
            if len(group) > 1:
                # Connect all transactions from same merchant
                for i, trans1 in enumerate(group):
                    for trans2 in group[i+1:]:
                        self.graph.add_edge(
                            trans1.id, trans2.id,
                            weight=0.8,
                            edge_type='merchant'
                        )
    
    def _add_amount_edges(self, transactions: List[Transaction], db: Session):
        """Add edges between transactions with similar amounts."""
        print("Adding amount edges...")
        
        for i, trans1 in enumerate(transactions):
            for j, trans2 in enumerate(transactions[i+1:], i+1):
                # Calculate amount similarity
                amount_diff = abs(trans1.amount - trans2.amount)
                max_amount = max(abs(trans1.amount), abs(trans2.amount))
                
                if max_amount > 0:
                    similarity = 1.0 - (amount_diff / max_amount)
                    
                    if similarity >= 0.9:  # Very similar amounts
                        self.graph.add_edge(
                            trans1.id, trans2.id,
                            weight=similarity,
                            edge_type='amount'
                        )
    
    def _prepare_graph_data(self, transactions: List[Transaction]):
        """Prepare graph data for GNN training."""
        print("Preparing graph data for GNN...")
        
        # Create node feature matrix
        node_features = []
        node_ids = []
        
        for transaction in transactions:
            if transaction.id in self.graph.nodes:
                features = self.graph.nodes[transaction.id]['features']
                node_features.append(features)
                node_ids.append(transaction.id)
        
        # Create edge index
        edge_index = []
        edge_weights = []
        
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for edge in self.graph.edges(data=True):
            if edge[0] in id_to_idx and edge[1] in id_to_idx:
                edge_index.append([id_to_idx[edge[0]], id_to_idx[edge[1]]])
                edge_index.append([id_to_idx[edge[1]], id_to_idx[edge[0]]])  # Undirected
                weight = edge[2].get('weight', 1.0)
                edge_weights.extend([weight, weight])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        self.graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        
        print(f"Graph data prepared: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    def train_advanced_gnn(self, transactions: List[Transaction]):
        """Train the advanced GNN for multiple tasks."""
        if not self.graph_data:
            print("No graph data available for training")
            return
        
        print("Training advanced GNN...")
        
        # Initialize model
        input_dim = self.graph_data.x.shape[1]
        self.gnn_model = AdvancedTransactionGNN(input_dim)
        
        # Simple self-supervised training (node reconstruction)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        for epoch in range(50):
            self.gnn_model.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gnn_model(self.graph_data.x, self.graph_data.edge_index)
            
            # Self-supervised loss (reconstruct node features)
            loss = F.mse_loss(embeddings, self.graph_data.x[:, :embeddings.shape[1]])
            
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("GNN training completed")
    
    def get_gnn_insights(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Use trained GNN to get insights about transactions."""
        if not self.gnn_model or not self.graph_data:
            return {"error": "GNN model not available"}
        
        self.gnn_model.eval()
        with torch.no_grad():
            # Get embeddings
            embeddings = self.gnn_model(self.graph_data.x, self.graph_data.edge_index, task='embedding')
            
            # Get anomaly scores
            anomaly_scores = self.gnn_model(self.graph_data.x, self.graph_data.edge_index, task='anomaly')
            
            # Get cluster predictions
            cluster_probs = self.gnn_model(self.graph_data.x, self.graph_data.edge_index, task='cluster')
            
            # Convert to numpy
            embeddings_np = embeddings.numpy()
            anomaly_scores_np = anomaly_scores.numpy().flatten()
            cluster_probs_np = cluster_probs.numpy()
            
            # Find anomalies (top 5% highest scores)
            anomaly_threshold = np.percentile(anomaly_scores_np, 95)
            anomaly_indices = np.where(anomaly_scores_np >= anomaly_threshold)[0]
            
            # Get cluster assignments
            cluster_assignments = np.argmax(cluster_probs_np, axis=1)
            
            return {
                "embeddings": embeddings_np,
                "anomaly_scores": anomaly_scores_np,
                "anomaly_threshold": anomaly_threshold,
                "anomaly_indices": anomaly_indices.tolist(),
                "cluster_assignments": cluster_assignments.tolist(),
                "cluster_probs": cluster_probs_np
            }
    
    def create_interactive_graph_visualization(self, transactions: List[Transaction], 
                                             insights: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create interactive graph visualization using Plotly."""
        if not self.graph.nodes:
            return go.Figure()
        
        # Get node positions using spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node_id in self.graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            transaction = self.graph.nodes[node_id]['transaction']
            node_text.append(f"ID: {transaction.id}<br>"
                           f"Amount: ${transaction.amount:.2f}<br>"
                           f"Date: {transaction.date.strftime('%Y-%m-%d')}<br>"
                           f"Merchant: {transaction.merchant or 'Unknown'}<br>"
                           f"Type: {transaction.transaction_type or 'Unknown'}")
            
            # Color based on amount (red for expenses, green for income)
            if transaction.amount > 0:
                node_color.append('green')
            else:
                node_color.append('red')
            
            # Size based on absolute amount
            node_size.append(min(max(abs(transaction.amount) / 100, 5), 50))
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"Type: {edge[2].get('edge_type', 'unknown')}, "
                           f"Weight: {edge[2].get('weight', 1.0):.3f}")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='lightgray'),
            hoverinfo='none',
            name='Edges'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='black'),
                opacity=0.8
            ),
            text=node_text,
            hoverinfo='text',
            name='Transactions'
        ))
        
        # Highlight anomalies if insights are provided
        if insights and 'anomaly_indices' in insights:
            anomaly_x = [node_x[i] for i in insights['anomaly_indices']]
            anomaly_y = [node_y[i] for i in insights['anomaly_indices']]
            
            fig.add_trace(go.Scatter(
                x=anomaly_x, y=anomaly_y,
                mode='markers',
                marker=dict(
                    size=15,
                    color='orange',
                    symbol='star',
                    line=dict(width=3, color='red')
                ),
                name='Anomalies',
                hoverinfo='text',
                text=[node_text[i] for i in insights['anomaly_indices']]
            ))
        
        # Update layout
        fig.update_layout(
            title="Interactive Transaction Graph",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for transaction details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_spending_pattern_visualization(self, transactions: List[Transaction]) -> go.Figure:
        """Create interactive spending pattern visualization."""
        # Convert to DataFrame for easier manipulation
        data = []
        for transaction in transactions:
            data.append({
                'date': transaction.date,
                'amount': transaction.amount,
                'merchant': transaction.merchant or 'Unknown',
                'type': transaction.transaction_type or 'Unknown',
                'is_expense': transaction.amount < 0
            })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spending Over Time', 'Spending by Category', 
                          'Top Merchants', 'Income vs Expenses'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Spending over time
        daily_spending = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=daily_spending['date'],
                y=daily_spending['amount'],
                mode='lines+markers',
                name='Daily Net Amount',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 2. Spending by category
        category_spending = df.groupby('type')['amount'].sum().abs().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=category_spending.index,
                y=category_spending.values,
                name='By Category',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Top merchants
        merchant_spending = df[df['is_expense']].groupby('merchant')['amount'].sum().abs().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=merchant_spending.values,
                y=merchant_spending.index,
                orientation='h',
                name='Top Merchants',
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        # 4. Income vs Expenses
        income = df[~df['is_expense']]['amount'].sum()
        expenses = df[df['is_expense']]['amount'].sum()
        fig.add_trace(
            go.Bar(
                x=['Income', 'Expenses'],
                y=[income, abs(expenses)],
                name='Income vs Expenses',
                marker_color=['green', 'red']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Interactive Spending Pattern Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if not self.graph.nodes:
            return {}
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        # Add centrality measures for top nodes
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            stats['most_central_nodes'] = sorted(
                degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            stats['bridge_nodes'] = sorted(
                betweenness_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        
        return stats
