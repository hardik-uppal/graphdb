import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from .config import Config
from .models import Transaction, GraphEdge, TransactionCluster
from .embedding_service import EmbeddingService

class TransactionGNN(nn.Module):
    """Graph Neural Network for transaction analysis."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super(TransactionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GraphService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.graph = nx.Graph()
        self.gnn_model = None
        
    def build_graph_from_transactions(self, transactions: List[Transaction], db: Session):
        """Build graph from transactions with multiple edge types."""
        print("Building graph from transactions...")
        
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes (transactions)
        for transaction in transactions:
            self.graph.add_node(
                transaction.id,
                transaction=transaction,
                amount=transaction.amount,
                date=transaction.date,
                type=transaction.transaction_type,
                merchant=transaction.merchant
            )
        
        # Add edges based on different criteria
        self._add_similarity_edges(transactions, db)
        self._add_temporal_edges(transactions, db)
        self._add_merchant_edges(transactions, db)
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
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
                    
                    # Store in database
                    edge = GraphEdge(
                        source_transaction_id=trans1.id,
                        target_transaction_id=trans2.id,
                        edge_type='similarity',
                        weight=similarity
                    )
                    db.add(edge)
        
        db.commit()
    
    def _add_temporal_edges(self, transactions: List[Transaction], db: Session):
        """Add edges based on temporal proximity."""
        print("Adding temporal edges...")
        
        # Sort by date
        sorted_transactions = sorted(transactions, key=lambda x: x.date)
        
        for i in range(len(sorted_transactions) - 1):
            trans1 = sorted_transactions[i]
            trans2 = sorted_transactions[i + 1]
            
            # Connect consecutive transactions within 3 days
            time_diff = abs((trans2.date - trans1.date).days)
            if time_diff <= 3:
                weight = 1.0 / (time_diff + 1)  # Higher weight for closer dates
                
                self.graph.add_edge(
                    trans1.id, trans2.id,
                    weight=weight,
                    edge_type='temporal'
                )
                
                edge = GraphEdge(
                    source_transaction_id=trans1.id,
                    target_transaction_id=trans2.id,
                    edge_type='temporal',
                    weight=weight
                )
                db.add(edge)
        
        db.commit()
    
    def _add_merchant_edges(self, transactions: List[Transaction], db: Session):
        """Add edges based on same merchant."""
        print("Adding merchant edges...")
        
        merchant_groups = {}
        for transaction in transactions:
            if transaction.merchant:
                merchant = transaction.merchant.strip()
                if merchant not in merchant_groups:
                    merchant_groups[merchant] = []
                merchant_groups[merchant].append(transaction)
        
        for merchant, group in merchant_groups.items():
            if len(group) > 1:
                # Connect all transactions from same merchant
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        trans1, trans2 = group[i], group[j]
                        
                        self.graph.add_edge(
                            trans1.id, trans2.id,
                            weight=1.0,
                            edge_type='merchant'
                        )
                        
                        edge = GraphEdge(
                            source_transaction_id=trans1.id,
                            target_transaction_id=trans2.id,
                            edge_type='merchant',
                            weight=1.0
                        )
                        db.add(edge)
        
        db.commit()
    
    def to_pytorch_geometric(self, transactions: List[Transaction]) -> Data:
        """Convert NetworkX graph to PyTorch Geometric format."""
        # Create node features from embeddings
        node_features = []
        node_mapping = {}
        
        for i, transaction in enumerate(transactions):
            node_mapping[transaction.id] = i
            if transaction.embedding:
                # Combine embedding with numerical features
                features = transaction.embedding + [
                    transaction.amount,
                    float(transaction.date.timestamp())
                ]
                node_features.append(features)
            else:
                # Zero features if no embedding
                features = [0.0] * (Config.EMBEDDING_DIMENSION + 2)
                node_features.append(features)
        
        # Create edge index and edge attributes
        edge_index = []
        edge_weights = []
        
        for edge in self.graph.edges(data=True):
            source_id, target_id, data = edge
            if source_id in node_mapping and target_id in node_mapping:
                source_idx = node_mapping[source_id]
                target_idx = node_mapping[target_id]
                
                edge_index.append([source_idx, target_idx])
                edge_index.append([target_idx, source_idx])  # Undirected
                
                weight = data.get('weight', 1.0)
                edge_weights.extend([weight, weight])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def train_gnn(self, transactions: List[Transaction], epochs=100):
        """Train the GNN model."""
        print("Training GNN model...")
        
        data = self.to_pytorch_geometric(transactions)
        
        # Initialize model
        input_dim = data.x.size(1)
        self.gnn_model = TransactionGNN(input_dim)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        # Simple reconstruction loss (autoencoder-style)
        self.gnn_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.gnn_model(data.x, data.edge_index)
            
            # Reconstruction loss (autoencoder style)
            # Use a simple reconstruction to original features
            decoder = nn.Linear(embeddings.size(1), data.x.size(1))
            reconstructed = decoder(embeddings)
            loss = F.mse_loss(reconstructed, data.x)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print("GNN training completed!")
    
    def get_node_embeddings(self, transactions: List[Transaction]) -> Dict[int, List[float]]:
        """Get enhanced node embeddings from trained GNN."""
        if self.gnn_model is None:
            print("GNN model not trained. Using original embeddings.")
            return {t.id: t.embedding for t in transactions if t.embedding}
        
        data = self.to_pytorch_geometric(transactions)
        
        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(data.x, data.edge_index)
        
        # Create mapping back to transaction IDs
        node_embeddings = {}
        for i, transaction in enumerate(transactions):
            node_embeddings[transaction.id] = embeddings[i].tolist()
        
        return node_embeddings
    
    def detect_transaction_clusters(self, transactions: List[Transaction], 
                                  db: Session, n_clusters=10) -> List[TransactionCluster]:
        """Detect transaction clusters using enhanced embeddings."""
        print(f"Detecting {n_clusters} transaction clusters...")
        
        # Get enhanced embeddings
        node_embeddings = self.get_node_embeddings(transactions)
        
        # Prepare data for clustering
        embedding_matrix = []
        transaction_ids = []
        
        for transaction in transactions:
            if transaction.id in node_embeddings:
                embedding_matrix.append(node_embeddings[transaction.id])
                transaction_ids.append(transaction.id)
        
        if not embedding_matrix:
            return []
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # Create cluster objects
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_transaction_ids = [
                transaction_ids[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            if cluster_transaction_ids:
                # Generate cluster description based on common patterns
                cluster_transactions = [
                    t for t in transactions if t.id in cluster_transaction_ids
                ]
                cluster_name, cluster_desc = self._generate_cluster_description(cluster_transactions)
                
                cluster = TransactionCluster(
                    name=cluster_name,
                    description=cluster_desc,
                    transaction_ids=cluster_transaction_ids,
                    centroid_embedding=kmeans.cluster_centers_[cluster_id].tolist()
                )
                
                db.add(cluster)
                clusters.append(cluster)
        
        db.commit()
        print(f"Created {len(clusters)} clusters")
        return clusters
    
    def _generate_cluster_description(self, transactions: List[Transaction]) -> Tuple[str, str]:
        """Generate description for a cluster of transactions."""
        if not transactions:
            return "Empty Cluster", "No transactions in this cluster"
        
        # Analyze common patterns
        merchants = [t.merchant for t in transactions if t.merchant]
        types = [t.transaction_type for t in transactions if t.transaction_type]
        amounts = [t.amount for t in transactions]
        
        # Most common merchant
        if merchants:
            most_common_merchant = max(set(merchants), key=merchants.count)
            if merchants.count(most_common_merchant) > len(transactions) * 0.5:
                return f"{most_common_merchant} Transactions", f"Transactions with {most_common_merchant}"
        
        # Most common type
        if types:
            most_common_type = max(set(types), key=types.count)
            avg_amount = np.mean([abs(a) for a in amounts])
            
            return f"{most_common_type} Cluster", f"Average amount: ${avg_amount:.2f}"
        
        return "Mixed Transactions", f"Cluster of {len(transactions)} transactions"
    
    def find_anomalous_transactions(self, transactions: List[Transaction], 
                                   threshold_percentile=95) -> List[Transaction]:
        """Find anomalous transactions based on graph structure."""
        if not self.graph.nodes():
            return []
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(self.graph)
        eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
        
        # Find transactions with unusual centrality scores
        centrality_scores = []
        for node_id in self.graph.nodes():
            score = betweenness[node_id] + eigenvector[node_id]
            centrality_scores.append((node_id, score))
        
        # Sort by centrality score
        centrality_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top percentile as anomalies
        threshold_index = int(len(centrality_scores) * (100 - threshold_percentile) / 100)
        anomalous_ids = [item[0] for item in centrality_scores[:threshold_index]]
        
        return [t for t in transactions if t.id in anomalous_ids]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.graph.nodes():
            return {}
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
