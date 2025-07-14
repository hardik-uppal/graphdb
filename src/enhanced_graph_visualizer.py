import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import colorsys
from collections import defaultdict
import math

class EnhancedGraphVisualizer:
    """Enhanced visualization for transaction graphs with better representations."""
    
    def __init__(self):
        self.color_palette = {
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
        
        self.edge_colors = {
            'similarity': '#1f77b4',    # Blue
            'temporal': '#ff7f0e',      # Orange
            'merchant': '#2ca02c',      # Green
            'pattern': '#9467bd'        # Purple
        }
    
    def create_interactive_graph(self, graph: nx.Graph, title: str = "Enhanced Transaction Graph") -> go.Figure:
        """Create an interactive graph visualization with enhanced node representations."""
        
        if not graph.nodes():
            return self._create_empty_graph_message()
        
        # Calculate layout
        pos = self._calculate_enhanced_layout(graph)
        
        # Prepare node traces
        node_traces = self._create_enhanced_node_traces(graph, pos)
        
        # Prepare edge traces
        edge_traces = self._create_enhanced_edge_traces(graph, pos)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size: transaction amount | Color: spending category | Connections: relationships",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor="left", yanchor="bottom",
                    font=dict(size=12, color="gray")
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _calculate_enhanced_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Calculate enhanced layout considering node categories and patterns."""
        
        # Use spring layout as base
        pos = nx.spring_layout(graph, k=3, iterations=50)
        
        # Group nodes by category for better visual organization
        category_groups = {}
        for node_id in graph.nodes():
            category = graph.nodes[node_id].get('category', 'Other')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(node_id)
        
        # Adjust positions to cluster similar categories
        num_categories = len(category_groups)
        if num_categories > 1:
            # Create category centers in a circle
            category_centers = {}
            for i, category in enumerate(category_groups.keys()):
                angle = 2 * np.pi * i / num_categories
                category_centers[category] = (np.cos(angle), np.sin(angle))
            
            # Adjust node positions toward their category centers
            for category, nodes in category_groups.items():
                center_x, center_y = category_centers[category]
                for node_id in nodes:
                    current_x, current_y = pos[node_id]
                    # Move 30% toward category center
                    pos[node_id] = (
                        current_x * 0.7 + center_x * 0.3,
                        current_y * 0.7 + center_y * 0.3
                    )
        
        return pos
    
    def _create_enhanced_node_traces(self, graph: nx.Graph, pos: Dict) -> List[go.Scatter]:
        """Create enhanced node traces with better visual representation."""
        
        # Group nodes by category for separate traces
        category_nodes = {}
        for node_id in graph.nodes():
            category = graph.nodes[node_id].get('category', 'Other')
            if category not in category_nodes:
                category_nodes[category] = {
                    'x': [], 'y': [], 'text': [], 'customdata': [],
                    'size': [], 'color': []
                }
            
            x, y = pos[node_id]
            node_data = graph.nodes[node_id]
            
            category_nodes[category]['x'].append(x)
            category_nodes[category]['y'].append(y)
            category_nodes[category]['text'].append(self._create_node_hover_text(node_data))
            category_nodes[category]['customdata'].append(node_id)
            category_nodes[category]['size'].append(node_data.get('size', 20))
            category_nodes[category]['color'].append(self.color_palette.get(category, '#708090'))
        
        # Create traces for each category
        traces = []
        for category, data in category_nodes.items():
            if not data['x']:  # Skip empty categories
                continue
                
            trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers',
                name=f"{category} ({len(data['x'])})",
                text=data['text'],
                customdata=data['customdata'],
                hoverinfo='text',
                hovertemplate='<b>%{text}</b><extra></extra>',
                marker=dict(
                    size=data['size'],
                    color=data['color'][0],  # Use category color
                    line=dict(width=2, color='white'),
                    opacity=0.8,
                    sizemode='diameter',
                    sizeref=2.*max(data['size'])/(40.**2),
                    sizemin=4
                )
            )
            traces.append(trace)
        
        return traces
    
    def _create_enhanced_edge_traces(self, graph: nx.Graph, pos: Dict) -> List[go.Scatter]:
        """Create enhanced edge traces with different styles for different relationships."""
        
        edge_groups = {}
        
        for edge in graph.edges(data=True):
            source, target, data = edge
            edge_type = data.get('edge_type', 'unknown')
            
            if edge_type not in edge_groups:
                edge_groups[edge_type] = {
                    'x': [], 'y': [], 'text': [], 'width': [], 'color': []
                }
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Add edge line
            edge_groups[edge_type]['x'].extend([x0, x1, None])
            edge_groups[edge_type]['y'].extend([y0, y1, None])
            edge_groups[edge_type]['text'].append(data.get('label', f"{edge_type} connection"))
            edge_groups[edge_type]['width'].append(data.get('width', 1))
            edge_groups[edge_type]['color'].append(self.edge_colors.get(edge_type, '#888888'))
        
        # Create traces for each edge type
        traces = []
        for edge_type, data in edge_groups.items():
            if not data['x']:  # Skip empty edge types
                continue
            
            trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='lines',
                name=f"{edge_type.title()} connections",
                line=dict(
                    width=2,
                    color=self.edge_colors.get(edge_type, '#888888')
                ),
                hoverinfo='skip',
                showlegend=True,
                opacity=0.6
            )
            traces.append(trace)
        
        return traces
    
    def _create_node_hover_text(self, node_data: Dict) -> str:
        """Create detailed hover text for nodes."""
        
        # Basic information
        label = node_data.get('label', 'Unknown Transaction')
        category = node_data.get('category', 'Other')
        merchant = node_data.get('merchant', 'Unknown')
        amount = node_data.get('amount', 0)
        date = node_data.get('date')
        
        # Additional insights
        spending_pattern = node_data.get('spending_pattern', 'Unknown')
        centrality_score = node_data.get('centrality_score', 0)
        tags = node_data.get('tags', [])
        
        # Format date
        date_str = date.strftime('%m/%d/%Y %I:%M %p') if date else 'Unknown date'
        
        # Build hover text
        hover_text = f"""
<b>{label}</b><br>
<b>Category:</b> {category}<br>
<b>Merchant:</b> {merchant}<br>
<b>Amount:</b> ${amount:.2f}<br>
<b>Date:</b> {date_str}<br>
<b>Pattern:</b> {spending_pattern}<br>
<b>Centrality:</b> {centrality_score:.3f}<br>
<b>Tags:</b> {', '.join(tags[:3])}{'...' if len(tags) > 3 else ''}
        """.strip()
        
        return hover_text
    
    def create_pattern_analysis_chart(self, patterns: List[Any]) -> go.Figure:
        """Create visualization for detected patterns."""
        
        if not patterns:
            fig = go.Figure()
            fig.add_annotation(
                text="No patterns detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Group patterns by type
        pattern_types = {}
        for pattern in patterns:
            ptype = pattern.pattern_type
            if ptype not in pattern_types:
                pattern_types[ptype] = []
            pattern_types[ptype].append(pattern)
        
        # Create subplot data
        pattern_names = []
        confidences = []
        transaction_counts = []
        pattern_type_labels = []
        colors = []
        
        color_map = {
            'recurring': '#2E8B57',
            'spending_burst': '#FF6347',
            'loyalty': '#4169E1',
            'amount_cluster': '#9932CC',
            'temporal': '#FF8C00'
        }
        
        for ptype, pattern_list in pattern_types.items():
            for pattern in pattern_list:
                pattern_names.append(pattern.pattern_name)
                confidences.append(pattern.confidence)
                transaction_counts.append(len(pattern.transactions))
                pattern_type_labels.append(ptype.replace('_', ' ').title())
                colors.append(color_map.get(ptype, '#708090'))
        
        # Create bubble chart
        fig = go.Figure(data=go.Scatter(
            x=confidences,
            y=pattern_type_labels,
            mode='markers',
            marker=dict(
                size=[count * 2 for count in transaction_counts],  # Size based on transaction count
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white'),
                sizemode='diameter',
                sizeref=2.*max(transaction_counts, default=1)/(40.**2),
                sizemin=8
            ),
            text=[f"{name}<br>Confidence: {conf:.2f}<br>Transactions: {count}" 
                  for name, conf, count in zip(pattern_names, confidences, transaction_counts)],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Detected Patterns'
        ))
        
        fig.update_layout(
            title='Detected Transaction Patterns',
            xaxis_title='Confidence Score',
            yaxis_title='Pattern Type',
            showlegend=False,
            margin=dict(l=100, r=20, t=50, b=50)
        )
        
        return fig
    
    def create_category_distribution_chart(self, category_distribution: Dict[str, int]) -> go.Figure:
        """Create category distribution visualization."""
        
        if not category_distribution:
            return self._create_empty_chart_message("No category data available")
        
        categories = list(category_distribution.keys())
        counts = list(category_distribution.values())
        colors = [self.color_palette.get(cat, '#708090') for cat in categories]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            marker_colors=colors,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Transaction Distribution by Category',
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def create_spending_patterns_chart(self, pattern_distribution: Dict[str, int]) -> go.Figure:
        """Create spending patterns distribution chart."""
        
        if not pattern_distribution:
            return self._create_empty_chart_message("No pattern data available")
        
        patterns = list(pattern_distribution.keys())
        counts = list(pattern_distribution.values())
        
        # Create horizontal bar chart
        fig = go.Figure(data=[go.Bar(
            x=counts,
            y=patterns,
            orientation='h',
            marker_color='#1f77b4',
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Distribution of Spending Patterns',
            xaxis_title='Number of Transactions',
            yaxis_title='Spending Pattern',
            margin=dict(l=150, r=20, t=50, b=50)
        )
        
        return fig
    
    def create_temporal_insights_chart(self, temporal_data: Dict[str, Any]) -> go.Figure:
        """Create temporal insights visualization."""
        
        weekend_count = temporal_data.get('weekend_transactions', 0)
        weekday_count = temporal_data.get('weekday_transactions', 0)
        
        if weekend_count == 0 and weekday_count == 0:
            return self._create_empty_chart_message("No temporal data available")
        
        # Create comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Weekday', 'Weekend'],
                y=[weekday_count, weekend_count],
                marker_color=['#1f77b4', '#ff7f0e'],
                text=[weekday_count, weekend_count],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Weekday vs Weekend Transaction Distribution',
            xaxis_title='Time Period',
            yaxis_title='Number of Transactions',
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        return fig
    
    def create_merchant_insights_chart(self, merchant_data: Dict[str, Any]) -> go.Figure:
        """Create merchant insights visualization."""
        
        top_by_frequency = merchant_data.get('top_by_frequency', [])
        top_by_amount = merchant_data.get('top_by_amount', [])
        
        if not top_by_frequency and not top_by_amount:
            return self._create_empty_chart_message("No merchant data available")
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Merchants by Frequency', 'Top Merchants by Amount'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top by frequency
        if top_by_frequency:
            merchants_freq = [item[0] for item in top_by_frequency]
            counts_freq = [item[1] for item in top_by_frequency]
            
            fig.add_trace(
                go.Bar(x=counts_freq, y=merchants_freq, orientation='h', 
                       name='Frequency', marker_color='#2ca02c'),
                row=1, col=1
            )
        
        # Top by amount
        if top_by_amount:
            merchants_amt = [item[0] for item in top_by_amount]
            amounts_amt = [item[1] for item in top_by_amount]
            
            fig.add_trace(
                go.Bar(x=amounts_amt, y=merchants_amt, orientation='h',
                       name='Amount', marker_color='#d62728'),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Top Merchants Analysis',
            showlegend=False,
            margin=dict(l=100, r=20, t=80, b=50)
        )
        
        return fig
    
    def create_enhanced_visualization(self, graph_service, 
                                    layout_type: str = "spring",
                                    color_by: str = "category",
                                    size_by: str = "amount",
                                    filter_patterns: List[str] = None) -> go.Figure:
        """Create enhanced graph visualization with intelligent layouts and coloring."""
        
        if not graph_service.graph.nodes():
            return self._create_empty_graph_message()
        
        # Get layout positions
        pos = self._calculate_layout(graph_service.graph, layout_type)
        
        # Prepare node data with enhanced attributes
        node_data = self._prepare_enhanced_node_data(graph_service, pos, color_by, size_by, filter_patterns)
        
        # Prepare edge data
        edge_data = self._prepare_enhanced_edge_data(graph_service, pos)
        
        # Create the figure
        fig = go.Figure()
        
        # Add edges with varying weights and colors
        self._add_enhanced_edges(fig, edge_data)
        
        # Add nodes with intelligent coloring and sizing
        fig.add_trace(go.Scatter(
            x=node_data['x'],
            y=node_data['y'],
            mode='markers+text',
            marker=dict(
                size=node_data['sizes'],
                color=node_data['colors'],
                line=dict(width=1, color='white'),
                opacity=0.8,
                colorscale='Viridis' if color_by == 'amount' else None,
                showscale=color_by == 'amount'
            ),
            text=node_data['labels'],
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            hovertext=node_data['hover_texts'],
            hoverinfo='text',
            showlegend=False,
            name='Transactions'
        ))
        
        # Create legend if applicable
        self._add_color_legend(fig, color_by, node_data.get('legend_data', {}))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Enhanced Transaction Graph - Colored by {color_by.title()}",
                font=dict(size=16)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"Nodes: {len(node_data['x'])} | Edges: {graph_service.graph.number_of_edges()}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def _calculate_layout(self, graph: nx.Graph, layout_type: str) -> Dict[int, Tuple[float, float]]:
        """Calculate node positions using various layout algorithms."""
        
        if layout_type == "spring":
            return nx.spring_layout(graph, k=3, iterations=50)
        elif layout_type == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(graph)
            except:
                return nx.spring_layout(graph, k=3, iterations=50)
        elif layout_type == "circular":
            return nx.circular_layout(graph)
        elif layout_type == "community":
            return self._community_based_layout(graph)
        else:
            return nx.spring_layout(graph, k=3, iterations=50)
    
    def _community_based_layout(self, graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
        """Create layout based on community structure."""
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(graph)
            
            # Group nodes by community
            from collections import defaultdict
            import math
            community_groups = defaultdict(list)
            for node, community in communities.items():
                community_groups[community].append(node)
            
            pos = {}
            num_communities = len(community_groups)
            
            for i, (community, nodes) in enumerate(community_groups.items()):
                # Position communities in a circle
                angle = 2 * math.pi * i / num_communities
                center_x = 3 * math.cos(angle)
                center_y = 3 * math.sin(angle)
                
                # Layout nodes within each community
                subgraph = graph.subgraph(nodes)
                if len(nodes) > 1:
                    sub_pos = nx.spring_layout(subgraph, center=(center_x, center_y), k=0.5)
                else:
                    sub_pos = {nodes[0]: (center_x, center_y)}
                
                pos.update(sub_pos)
            
            return pos
            
        except ImportError:
            print("Community layout requires python-louvain package, falling back to spring layout")
            return nx.spring_layout(graph, k=3, iterations=50)
    
    def _prepare_enhanced_node_data(self, graph_service, pos: Dict, 
                                  color_by: str, size_by: str, filter_patterns: List[str] = None) -> Dict:
        """Prepare enhanced node data for visualization."""
        
        node_data = {
            'x': [], 'y': [], 'colors': [], 'sizes': [], 
            'labels': [], 'hover_texts': [], 'legend_data': {}
        }
        
        # Color schemes
        color_schemes = {
            'category': {
                'Grocery': '#2E8B57', 'Restaurant': '#FF6347', 'Gas': '#4169E1',
                'Retail': '#9370DB', 'Healthcare': '#DC143C', 'Entertainment': '#FF1493',
                'Transportation': '#00CED1', 'Utilities': '#8B4513', 'Financial': '#DAA520',
                'Other': '#696969'
            },
            'pattern': {
                'Regular/Recurring': '#2E8B57', 'Frequent/Irregular': '#FFD700',
                'Occasional': '#87CEEB', 'Large/Unusual': '#FF6347', 
                'Infrequent': '#D3D3D3', 'Isolated': '#A9A9A9'
            }
        }
        
        for node_id in graph_service.graph.nodes():
            if node_id not in pos:
                continue
                
            node_attrs = graph_service.graph.nodes[node_id]
            
            # Apply filters if specified
            if filter_patterns and node_attrs.get('spending_pattern') not in filter_patterns:
                continue
            
            # Position
            x, y = pos[node_id]
            node_data['x'].append(x)
            node_data['y'].append(y)
            
            # Color based on scheme
            if color_by == 'category':
                category = node_attrs.get('category', 'Other')
                color = color_schemes['category'].get(category, '#696969')
            elif color_by == 'pattern':
                pattern = node_attrs.get('spending_pattern', 'Isolated')
                color = color_schemes['pattern'].get(pattern, '#A9A9A9')
            elif color_by == 'amount':
                # Use numerical value for amount-based coloring
                amount_bucket = node_attrs.get('amount_bucket', 'Small ($10-50)')
                if 'Micro' in amount_bucket:
                    color = 1
                elif 'Small' in amount_bucket:
                    color = 2
                elif 'Medium' in amount_bucket:
                    color = 3
                elif 'Large' in amount_bucket:
                    color = 4
                else:  # XLarge
                    color = 5
            else:
                color = '#87CEEB'
            
            node_data['colors'].append(color)
            
            # Size based on scheme
            if size_by == 'amount':
                amount_bucket = node_attrs.get('amount_bucket', 'Small ($10-50)')
                if 'Micro' in amount_bucket:
                    size = 8
                elif 'Small' in amount_bucket:
                    size = 12
                elif 'Medium' in amount_bucket:
                    size = 16
                elif 'Large' in amount_bucket:
                    size = 20
                else:  # XLarge
                    size = 25
            elif size_by == 'frequency':
                frequency = node_attrs.get('frequency_score', 0)
                size = max(8, min(25, 8 + frequency * 100))
            else:
                size = 12
            
            node_data['sizes'].append(size)
            
            # Label and hover text
            label = node_attrs.get('display_label', f'Transaction {node_id}')
            # Truncate label for display
            if '\n' in label:
                lines = label.split('\n')
                if len(lines[0]) > 12:
                    label = lines[0][:10] + '...'
                else:
                    label = lines[0]
            
            node_data['labels'].append(label)
            
            # Detailed hover text
            hover_text = f"<b>{node_attrs.get('merchant', 'Unknown')}</b><br>"
            hover_text += f"Category: {node_attrs.get('category', 'Unknown')}<br>"
            hover_text += f"Amount: {node_attrs.get('amount_bucket', 'Unknown')}<br>"
            hover_text += f"Pattern: {node_attrs.get('spending_pattern', 'Unknown')}<br>"
            hover_text += f"Time: {node_attrs.get('time_bucket', 'Unknown')}"
            
            node_data['hover_texts'].append(hover_text)
        
        return node_data
    
    def _prepare_enhanced_edge_data(self, graph_service, pos: Dict) -> Dict:
        """Prepare enhanced edge data for visualization."""
        
        edge_data = {'traces': []}
        
        # Group edges by type for different styling
        edge_types = {'similarity': [], 'temporal': [], 'merchant': [], 'pattern': []}
        
        for edge in graph_service.graph.edges(data=True):
            source, target, attrs = edge
            
            if source not in pos or target not in pos:
                continue
            
            edge_type = attrs.get('edge_type', 'unknown')
            weight = attrs.get('weight', 0.5)
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_info = {
                'x': [x0, x1, None],
                'y': [y0, y1, None],
                'weight': weight,
                'label': attrs.get('label', f'{edge_type} edge')
            }
            
            if edge_type in edge_types:
                edge_types[edge_type].append(edge_info)
        
        edge_data['edge_types'] = edge_types
        return edge_data
    
    def _add_enhanced_edges(self, fig: go.Figure, edge_data: Dict):
        """Add enhanced edges with different colors and weights."""
        
        edge_colors = {
            'similarity': 'rgba(255, 100, 100, 0.6)',
            'temporal': 'rgba(100, 255, 100, 0.6)',
            'merchant': 'rgba(100, 100, 255, 0.6)',
            'pattern': 'rgba(255, 100, 255, 0.6)'
        }
        
        edge_types = edge_data.get('edge_types', {})
        
        for edge_type, edges in edge_types.items():
            if not edges:
                continue
            
            x_coords = []
            y_coords = []
            
            for edge in edges:
                x_coords.extend(edge['x'])
                y_coords.extend(edge['y'])
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(
                    width=1,
                    color=edge_colors.get(edge_type, 'rgba(128,128,128,0.3)')
                ),
                hoverinfo='none',
                showlegend=True,
                name=f'{edge_type.title()} Connections',
                legendgroup=edge_type
            ))
    
    def _add_color_legend(self, fig: go.Figure, color_by: str, legend_data: Dict):
        """Add legend based on coloring scheme."""
        
        if color_by == 'category':
            categories = ['Grocery', 'Restaurant', 'Gas', 'Retail', 'Other']
            colors = ['#2E8B57', '#FF6347', '#4169E1', '#9370DB', '#696969']
            
            for category, color in zip(categories, colors):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=True,
                    name=category,
                    legendgroup='categories'
                ))
        
        elif color_by == 'pattern':
            patterns = ['Regular/Recurring', 'Frequent/Irregular', 'Occasional', 'Infrequent']
            colors = ['#2E8B57', '#FFD700', '#87CEEB', '#D3D3D3']
            
            for pattern, color in zip(patterns, colors):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=True,
                    name=pattern,
                    legendgroup='patterns'
                ))

    def _create_empty_graph_message(self) -> go.Figure:
        """Create an empty graph with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available.<br>Please build the graph first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def _create_empty_chart_message(self, message: str) -> go.Figure:
        """Create an empty chart with custom message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        return fig
