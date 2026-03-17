import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional
import math

class ClusterGraphRenderer:
    """Renders clusters as draggable nodes."""
    
    def __init__(self):
        self.node_size = 12
        self.cluster_padding = 0.3
        self.connection_threshold = 0.02
        self.node_spacing = 0.2
        
    def create_cluster_graph(self, clusters: Dict, graph_data, 
                           cluster_positions: Optional[Dict] = None) -> go.Figure:
        """Create cluster graph with proper draggable centers."""
        
        if not clusters:
            return self._create_empty_graph()
        
        # Calculate cluster positions
        positions = self._calculate_cluster_positions(clusters, cluster_positions)
        
        # Create figure
        fig = go.Figure()
        
        # Add cluster connections first (so they appear behind)
        self._add_cluster_connections(fig, clusters, positions, graph_data)
        
        # Add each cluster 
        for cluster_id, cluster_data in clusters.items():
            self._add_cluster(fig, cluster_id, cluster_data, positions[cluster_id])
        
        # Enable INDIVIDUAL dragging with layout settings
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-8, 8]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-6, 6]
            ),
            plot_bgcolor='#fafafa',
            paper_bgcolor='#fafafa',
            dragmode='pan',  # Allow panning for navigation
            # THIS IS THE KEY: uirevision preserves drag state
            uirevision='true'
        )
        
        return fig
    
    def _add_cluster(self, fig: go.Figure, cluster_id: str, 
                    cluster_data: Dict, position: Dict):
        """Add a single draggable cluster."""
        nodes = cluster_data.get('nodes', [])
        color = cluster_data.get('color', '#3b82f6')
        name = cluster_data.get('name', f'Cluster {cluster_id[:6]}')
        
        if nodes:
            # Filled cluster 
            self._add_cluster_background(fig, cluster_id, cluster_data, position, nodes)
            
            # Add draggable center point
            fig.add_trace(go.Scatter(
                x=[position['x']],
                y=[position['y']],
                mode='markers',
                marker=dict(
                    size=25,
                    color=color,
                    opacity=0.8,
                    symbol='circle',
                    line=dict(width=3, color='white')
                ),
                customdata=[cluster_id],
                hoverinfo='text',
                hovertext=f"{name} ({len(nodes)} features)<br>Drag to move • Click to add feature",
                showlegend=False,
                name=f'cluster_{cluster_id}',
                # Make draggable
                xaxis='x',
                yaxis='y',
            ))
        else:
            # Empty cluster
            fig.add_trace(go.Scatter(
                x=[position['x']],
                y=[position['y']],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color='white',
                    opacity=1.0,
                    symbol='circle',
                    line=dict(width=3, color=color)
                ),
                text=['+'],
                textfont=dict(size=16, color=color),
                textposition='middle center',
                customdata=[cluster_id],
                hoverinfo='text',
                hovertext=f"{name}<br>Empty cluster<br>Drag to move • Click to add feature",
                showlegend=False,
                name=f'cluster_{cluster_id}',
                # Make draggable
                xaxis='x',
                yaxis='y',
            ))
    
    def _add_cluster_background(self, fig: go.Figure, cluster_id: str, 
                              cluster_data: Dict, cluster_center: Dict, nodes):
        """Add background elements for filled cluster."""
        color = cluster_data.get('color', '#3b82f6')
        
        # Calculate node positions
        node_positions = self._calculate_node_positions(nodes, cluster_center)
        
        if not node_positions:
            return
        
        # Draw rectangle around nodes
        min_x = min(pos['x'] for pos in node_positions) - self.cluster_padding
        max_x = max(pos['x'] for pos in node_positions) + self.cluster_padding
        min_y = min(pos['y'] for pos in node_positions) - self.cluster_padding
        max_y = max(pos['y'] for pos in node_positions) + self.cluster_padding
        
        # Background rectangle
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y,
            x1=max_x, y1=max_y,
            line=dict(color=color, width=2),
            fillcolor=f'rgba({self._hex_to_rgba(color)}, 0.1)',
            layer="below"
        )
        
        # Feature nodes
        x_coords = [pos['x'] for pos in node_positions]
        y_coords = [pos['y'] for pos in node_positions]
        
        hover_texts = []
        for node in nodes:
            hover_texts.append(f"L{node.get('layer', '?')}.{node.get('feature_idx', '?')}")
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=self.node_size,
                color=color,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            showlegend=False
        ))
    
    def _calculate_node_positions(self, nodes: List, cluster_center: Dict) -> List[Dict]:
        """Calculate node positions in a simple grid."""
        node_count = len(nodes)
        if node_count == 0:
            return []
        
        positions = []
        
        if node_count == 1:
            positions.append(cluster_center)
        elif node_count <= 4:
            # Line arrangement
            for i in range(node_count):
                offset = (i - (node_count - 1) / 2) * self.node_spacing
                positions.append({
                    'x': cluster_center['x'] + offset,
                    'y': cluster_center['y']
                })
        else:
            # Grid arrangement
            cols = math.ceil(math.sqrt(node_count))
            rows = math.ceil(node_count / cols)
            
            for i in range(node_count):
                row = i // cols
                col = i % cols
                
                offset_x = (col - (cols - 1) / 2) * self.node_spacing
                offset_y = (row - (rows - 1) / 2) * self.node_spacing
                
                positions.append({
                    'x': cluster_center['x'] + offset_x,
                    'y': cluster_center['y'] + offset_y
                })
        
        return positions
    
    def _hex_to_rgba(self, hex_color: str) -> str:
        """Convert hex color to rgba."""
        hex_color = hex_color.lstrip('#')
        return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
    
    def _add_cluster_connections(self, fig: go.Figure, clusters: Dict, 
                               positions: Dict, graph_data):
        """Add connections between clusters."""
        cluster_ids = list(clusters.keys())
        
        for i, cluster_id_1 in enumerate(cluster_ids):
            for j, cluster_id_2 in enumerate(cluster_ids):
                if i < j:
                    if (clusters[cluster_id_1].get('nodes') and 
                        clusters[cluster_id_2].get('nodes')):
                        
                        strength = self._calculate_connection_strength(
                            clusters[cluster_id_1], clusters[cluster_id_2], graph_data
                        )
                        
                        if strength > self.connection_threshold:
                            pos1 = positions[cluster_id_1]
                            pos2 = positions[cluster_id_2]
                            
                            fig.add_trace(go.Scatter(
                                x=[pos1['x'], pos2['x']],
                                y=[pos1['y'], pos2['y']],
                                mode='lines',
                                line=dict(
                                    width=max(1, strength * 20),
                                    color=f'rgba(150, 150, 150, {min(0.8, strength * 4)})'
                                ),
                                hoverinfo='text',
                                hovertext=f"Connection: {strength:.3f}",
                                showlegend=False
                            ))
    
    def _calculate_connection_strength(self, cluster1: Dict, cluster2: Dict, 
                                     graph_data) -> float:
        """Calculate connection strength."""
        nodes1 = cluster1.get('nodes', [])
        nodes2 = cluster2.get('nodes', [])
        
        if not nodes1 or not nodes2:
            return 0
        
        total_weight = 0
        connection_count = 0
        adj_matrix = graph_data.adjacency_matrix
        
        for node1 in nodes1:
            for node2 in nodes2:
                idx1 = node1.get('node_index', -1)
                idx2 = node2.get('node_index', -1)
                
                if (0 <= idx1 < len(adj_matrix) and 
                    0 <= idx2 < len(adj_matrix[0])):
                    weight = abs(adj_matrix[idx1, idx2])
                    if weight > 0:
                        total_weight += weight
                        connection_count += 1
        
        return total_weight / max(connection_count, 1) if connection_count > 0 else 0
    
    def _calculate_cluster_positions(self, clusters: Dict, 
                                   cluster_positions: Optional[Dict] = None) -> Dict:
        """Calculate cluster positions."""
        cluster_ids = list(clusters.keys())
        
        if cluster_positions:
            positions = {}
            for cluster_id in cluster_ids:
                if cluster_id in cluster_positions:
                    positions[cluster_id] = cluster_positions[cluster_id]
                else:
                    positions[cluster_id] = self._find_free_position(positions)
            return positions
        
        # Initial positioning
        positions = {}
        if len(cluster_ids) == 1:
            positions[cluster_ids[0]] = {'x': 0, 'y': 0}
        else:
            for i, cluster_id in enumerate(cluster_ids):
                angle = 2 * math.pi * i / len(cluster_ids)
                radius = 3
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions[cluster_id] = {'x': x, 'y': y}
        
        return positions
    
    def _find_free_position(self, existing_positions: Dict) -> Dict:
        """Find free position for new cluster."""
        if not existing_positions:
            return {'x': 0, 'y': 0}
        
        for radius in [2, 3, 4, 5]:
            for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                
                is_free = True
                for pos in existing_positions.values():
                    if math.sqrt((x - pos['x'])**2 + (y - pos['y'])**2) < 1.5:
                        is_free = False
                        break
                
                if is_free:
                    return {'x': x, 'y': y}
        
        return {'x': len(existing_positions) * 2, 'y': 0}
    
    def _create_empty_graph(self) -> go.Figure:
        """Create empty graph."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0, y=0,
            text="Click + Cluster to start grouping features",
            showarrow=False,
            font=dict(size=14, color='#9ca3af')
        )
        
        fig.update_layout(
            xaxis=dict(range=[-5, 5], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#fafafa',
            paper_bgcolor='#fafafa',
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        
        return fig
