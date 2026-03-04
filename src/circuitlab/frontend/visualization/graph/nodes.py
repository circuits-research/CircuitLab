import plotly.graph_objects as go
from typing import List, Set, Optional

from ...data.models import FeatureNode
from ...config.settings import GraphConfig

class NodeRenderer:
    """Handles rendering of graph nodes."""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        # Node sizes
        self.normal_node_size = 16
        self.selected_node_size = 22
        self.highlighted_node_size = 19
        self.description_node_size = 16  # For single-clicked nodes with descriptions
        
        # Get colors with fallbacks
        self.normal_node_color = getattr(config, 'normal_node_color', '#f8f9fa')  # Light gray/white fill
        self.selected_node_color = getattr(config, 'selected_node_color', '#EF4444')
        self.highlighted_node_color = getattr(config, 'highlighted_node_color', '#F59E0B')
        self.description_node_color = '#10B981'  # Green for nodes with descriptions
        self.intersection_node_color = '#DC2626'  # Red for intersection nodes
    
    def create_node_trace(self, nodes: List[FeatureNode],
                         selected_feature_id: Optional[int] = None,
                         highlighted_nodes: Set[int] = None,
                         nodes_with_descriptions: Set[int] = None,
                         node_to_cluster: dict = None,
                         cluster_highlighted_nodes: Set[int] = None,
                         intersection_nodes: Set[int] = None) -> List[go.Scatter]:
        """Create separate traces for embedding nodes (squares) and regular nodes (circles)."""
        if not nodes:
            return [go.Scatter(x=[], y=[], mode='markers')]

        highlighted_nodes = highlighted_nodes or set()
        nodes_with_descriptions = nodes_with_descriptions or set()
        node_to_cluster = node_to_cluster or {}
        cluster_highlighted_nodes = cluster_highlighted_nodes or set()
        intersection_nodes = intersection_nodes or set()

        # Separate embedding nodes from regular nodes
        embedding_nodes = []
        regular_nodes = []
        embedding_indices = []
        regular_indices = []

        for i, node in enumerate(nodes):
            if node.layer == "embedding":
                embedding_nodes.append(node)
                embedding_indices.append(i)
            else:
                regular_nodes.append(node)
                regular_indices.append(i)

        traces = []

        # Create embedding nodes trace (small squares)
        if embedding_nodes:
            embedding_trace = self._create_embedding_trace(
                embedding_nodes, embedding_indices, highlighted_nodes,
                nodes_with_descriptions, intersection_nodes
            )
            traces.append(embedding_trace)

        # Create regular nodes trace (circles)
        if regular_nodes:
            regular_trace = self._create_regular_node_trace(
                regular_nodes, regular_indices, selected_feature_id,
                highlighted_nodes, nodes_with_descriptions, node_to_cluster,
                cluster_highlighted_nodes, intersection_nodes
            )
            traces.append(regular_trace)

        return traces

    def _create_embedding_trace(self, embedding_nodes: List[FeatureNode], embedding_indices: List[int],
                               highlighted_nodes: Set[int], nodes_with_descriptions: Set[int],
                               intersection_nodes: Set[int]) -> go.Scatter:
        """Create trace for embedding nodes as small squares."""
        x_coords = [node.x for node in embedding_nodes]
        y_coords = [node.y for node in embedding_nodes]

        # Build hover text for embedding nodes
        hover_texts = []
        for node in embedding_nodes:
            hover_info = [
                f"Embedding Token: {node.token}",
                f"Position: {node.pos}",
                f"Description: {node.description}"
            ]
            hover_texts.append("<br>".join(hover_info))

        # Embedding nodes are small and gray
        sizes = [10] * len(embedding_nodes)  # Small squares
        colors = ['#9CA3AF'] * len(embedding_nodes)  # Gray color

        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol='square',  # Square symbols for embeddings
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False,
            name='Embedding Tokens'
        )

    def _create_regular_node_trace(self, regular_nodes: List[FeatureNode], regular_indices: List[int],
                                  selected_feature_id: Optional[int], highlighted_nodes: Set[int],
                                  nodes_with_descriptions: Set[int], node_to_cluster: dict,
                                  cluster_highlighted_nodes: Set[int], intersection_nodes: Set[int]) -> go.Scatter:
        """Create trace for regular feature nodes as circles."""
        x_coords = [node.x for node in regular_nodes]
        y_coords = [node.y for node in regular_nodes]

        # Build hover text with frequency information
        hover_texts = []
        for node in regular_nodes:
            hover_info = [
                f"Layer: {node.layer}",
                f"Position: {node.pos}",
                f"Feature: {node.feature_idx}",
                f"Token: {node.token}"
            ]

            if node.frequency is not None:
                hover_info.append(f"Frequency: {node.frequency:.3f}")
            else:
                hover_info.append("Frequency: N/A")

            if hasattr(node, 'description') and node.description:
                # Truncate long descriptions for hover
                desc = node.description
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                hover_info.append(f"Description: {desc}")

            hover_texts.append("<br>".join(hover_info))

        # Calculate node sizes and colors
        sizes = []
        colors = []

        for i, original_idx in enumerate(regular_indices):
            if original_idx == selected_feature_id:
                # Double-clicked node (red, largest)
                sizes.append(self.selected_node_size)
                colors.append(self.selected_node_color)
            elif original_idx in intersection_nodes:
                # Intersection node (red, but smaller than selected)
                sizes.append(self.highlighted_node_size)
                colors.append(self.intersection_node_color)
            elif original_idx in highlighted_nodes:
                # Connected to double-clicked node (orange)
                sizes.append(self.highlighted_node_size)
                colors.append(self.highlighted_node_color)
            elif original_idx in nodes_with_descriptions:
                # Single-clicked node with description (green)
                sizes.append(self.description_node_size)
                colors.append(self.description_node_color)
            elif original_idx in cluster_highlighted_nodes:
                # Node in selected cluster - use cluster color with larger size
                sizes.append(self.highlighted_node_size)
                cluster_color = node_to_cluster.get(original_idx, self.highlighted_node_color)
                colors.append(cluster_color)
            else:
                # Normal node - ALWAYS use default blue color, regardless of cluster membership
                sizes.append(self.normal_node_size)
                colors.append(self.normal_node_color)  # Always blue, never cluster color

        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol='circle',  # Circle symbols for regular features
                line=dict(width=2, color='black'),  # Black borders
                opacity=1.0
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False,
            name='Feature Nodes'
        )

    def create_node_labels(self, nodes: List[FeatureNode]) -> List[go.Scatter]:
        """Return empty list - no labels by default."""
        return []
    
    def get_node_style(self, node: FeatureNode, is_selected: bool = False, 
                      is_highlighted: bool = False, has_description: bool = False) -> dict:
        """Get styling for individual nodes."""
        if is_selected:
            return {
                'size': self.selected_node_size,
                'color': self.selected_node_color,
                'line_width': 2
            }
        elif is_highlighted:
            return {
                'size': self.highlighted_node_size,
                'color': self.highlighted_node_color,
                'line_width': 1.5
            }
        elif has_description:
            return {
                'size': self.description_node_size,
                'color': self.description_node_color,
                'line_width': 1.5
            }
        else:
            return {
                'size': self.normal_node_size,
                'color': self.normal_node_color,
                'line_width': 1
            }
