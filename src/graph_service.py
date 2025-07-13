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
    
    def detect_communities(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect communities in the transaction graph using multiple algorithms."""
        if not self.graph.nodes():
            return {"error": "No graph data available"}
        
        results = {}
        
        try:
            # Louvain community detection
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Organize communities
            communities = {}
            for node_id, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node_id)
            
            # Get community statistics
            community_stats = []
            for comm_id, node_ids in communities.items():
                comm_transactions = [t for t in transactions if t.id in node_ids]
                
                if comm_transactions:
                    total_amount = sum(abs(t.amount) for t in comm_transactions)
                    avg_amount = total_amount / len(comm_transactions)
                    
                    merchants = [t.merchant for t in comm_transactions if t.merchant]
                    top_merchant = max(set(merchants), key=merchants.count) if merchants else "Unknown"
                    
                    community_stats.append({
                        'community_id': comm_id,
                        'size': len(node_ids),
                        'transaction_ids': node_ids,
                        'total_amount': total_amount,
                        'avg_amount': avg_amount,
                        'top_merchant': top_merchant,
                        'transaction_types': list(set(t.transaction_type for t in comm_transactions))
                    })
            
            results['louvain'] = {
                'num_communities': len(communities),
                'modularity': community_louvain.modularity(partition, self.graph),
                'communities': community_stats
            }
            
        except ImportError:
            results['louvain'] = {"error": "python-louvain not installed"}
        except Exception as e:
            results['louvain'] = {"error": str(e)}
        
        # Connected components (simpler alternative)
        try:
            connected_components = list(nx.connected_components(self.graph))
            results['connected_components'] = {
                'num_components': len(connected_components),
                'components': [list(comp) for comp in connected_components]
            }
        except Exception as e:
            results['connected_components'] = {"error": str(e)}
        
        return results
    
    def calculate_centrality_measures(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Calculate various centrality measures for transactions."""
        if not self.graph.nodes():
            return {"error": "No graph data available"}
        
        centrality_results = {}
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # PageRank
            pagerank = nx.pagerank(self.graph)
            
            # Eigenvector centrality (if graph is connected)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph)
            except nx.NetworkXError:
                eigenvector_centrality = {}
            
            # Create transaction-level results
            transaction_centralities = []
            for transaction in transactions:
                if transaction.id in degree_centrality:
                    transaction_centralities.append({
                        'transaction_id': transaction.id,
                        'merchant': transaction.merchant,
                        'amount': transaction.amount,
                        'date': transaction.date.isoformat(),
                        'degree_centrality': degree_centrality[transaction.id],
                        'betweenness_centrality': betweenness_centrality.get(transaction.id, 0),
                        'closeness_centrality': closeness_centrality.get(transaction.id, 0),
                        'pagerank': pagerank.get(transaction.id, 0),
                        'eigenvector_centrality': eigenvector_centrality.get(transaction.id, 0)
                    })
            
            # Sort by different centrality measures
            centrality_results = {
                'transaction_centralities': transaction_centralities,
                'top_by_degree': sorted(transaction_centralities, 
                                      key=lambda x: x['degree_centrality'], reverse=True)[:10],
                'top_by_betweenness': sorted(transaction_centralities, 
                                           key=lambda x: x['betweenness_centrality'], reverse=True)[:10],
                'top_by_pagerank': sorted(transaction_centralities, 
                                        key=lambda x: x['pagerank'], reverse=True)[:10]
            }
            
        except Exception as e:
            centrality_results = {"error": str(e)}
        
        return centrality_results
    
    def find_shortest_paths(self, source_transaction_id: int, target_transaction_id: int) -> Dict[str, Any]:
        """Find shortest paths between two transactions."""
        if not self.graph.has_node(source_transaction_id) or not self.graph.has_node(target_transaction_id):
            return {"error": "One or both transactions not found in graph"}
        
        try:
            # Find shortest path
            path = nx.shortest_path(self.graph, source_transaction_id, target_transaction_id)
            path_length = nx.shortest_path_length(self.graph, source_transaction_id, target_transaction_id)
            
            # Get path details
            path_details = []
            for i, node_id in enumerate(path):
                node_data = self.graph.nodes[node_id]
                path_details.append({
                    'position': i,
                    'transaction_id': node_id,
                    'merchant': node_data.get('merchant'),
                    'amount': node_data.get('amount'),
                    'date': node_data.get('date').isoformat() if node_data.get('date') else None
                })
                
                # Add edge information
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_data = self.graph.edges[node_id, next_node]
                    path_details[i]['edge_to_next'] = {
                        'type': edge_data.get('edge_type'),
                        'weight': edge_data.get('weight')
                    }
            
            return {
                'path': path,
                'path_length': path_length,
                'path_details': path_details
            }
            
        except nx.NetworkXNoPath:
            return {"error": "No path found between transactions"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_transaction_neighborhood(self, transaction_id: int, radius: int = 2) -> Dict[str, Any]:
        """Analyze the neighborhood of a specific transaction."""
        if not self.graph.has_node(transaction_id):
            return {"error": "Transaction not found in graph"}
        
        try:
            # Get neighbors at different distances
            neighbors_by_distance = {}
            for distance in range(1, radius + 1):
                neighbors = set()
                for node in self.graph.nodes():
                    try:
                        path_length = nx.shortest_path_length(self.graph, transaction_id, node)
                        if path_length == distance:
                            neighbors.add(node)
                    except nx.NetworkXNoPath:
                        pass
                neighbors_by_distance[distance] = list(neighbors)
            
            # Get subgraph
            all_neighbors = {transaction_id}
            for neighbors in neighbors_by_distance.values():
                all_neighbors.update(neighbors)
            
            subgraph = self.graph.subgraph(all_neighbors)
            
            # Calculate local metrics
            local_clustering = nx.clustering(subgraph, transaction_id)
            local_degree = self.graph.degree(transaction_id)
            
            return {
                'transaction_id': transaction_id,
                'neighbors_by_distance': neighbors_by_distance,
                'total_neighbors': len(all_neighbors) - 1,
                'local_clustering_coefficient': local_clustering,
                'degree': local_degree,
                'subgraph_stats': {
                    'nodes': subgraph.number_of_nodes(),
                    'edges': subgraph.number_of_edges(),
                    'density': nx.density(subgraph)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def detect_graph_anomalies(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect anomalies based on graph structure and centrality measures."""
        if not self.graph.nodes():
            return []
        
        anomalies = []
        
        try:
            # Calculate centrality measures
            centrality_data = self.calculate_centrality_measures(transactions)
            
            if 'error' in centrality_data:
                return []
            
            transaction_centralities = centrality_data['transaction_centralities']
            
            # Define thresholds (top 5% for each measure)
            def get_threshold(values, percentile=95):
                return np.percentile(values, percentile)
            
            degree_threshold = get_threshold([t['degree_centrality'] for t in transaction_centralities])
            betweenness_threshold = get_threshold([t['betweenness_centrality'] for t in transaction_centralities])
            pagerank_threshold = get_threshold([t['pagerank'] for t in transaction_centralities])
            
            # Find anomalies
            for t_data in transaction_centralities:
                anomaly_reasons = []
                
                # High degree centrality (many connections)
                if t_data['degree_centrality'] > degree_threshold:
                    anomaly_reasons.append(f"High degree centrality ({t_data['degree_centrality']:.3f})")
                
                # High betweenness centrality (bridge between groups)
                if t_data['betweenness_centrality'] > betweenness_threshold:
                    anomaly_reasons.append(f"High betweenness centrality ({t_data['betweenness_centrality']:.3f})")
                
                # High PageRank (influential)
                if t_data['pagerank'] > pagerank_threshold:
                    anomaly_reasons.append(f"High PageRank ({t_data['pagerank']:.3f})")
                
                # Check for unusual amount-centrality combinations
                amount_abs = abs(t_data['amount'])
                if amount_abs > 1000 and t_data['degree_centrality'] > degree_threshold:
                    anomaly_reasons.append(f"High amount (${amount_abs:.2f}) with high connectivity")
                
                if anomaly_reasons:
                    anomalies.append({
                        'transaction_id': t_data['transaction_id'],
                        'merchant': t_data['merchant'],
                        'amount': t_data['amount'],
                        'date': t_data['date'],
                        'anomaly_reasons': anomaly_reasons,
                        'centrality_scores': {
                            'degree': t_data['degree_centrality'],
                            'betweenness': t_data['betweenness_centrality'],
                            'pagerank': t_data['pagerank']
                        }
                    })
            
            # Sort by number of anomaly reasons
            anomalies.sort(key=lambda x: len(x['anomaly_reasons']), reverse=True)
            
        except Exception as e:
            return [{"error": str(e)}]
        
        return anomalies
