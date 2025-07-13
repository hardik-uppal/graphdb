import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import colorsys

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
