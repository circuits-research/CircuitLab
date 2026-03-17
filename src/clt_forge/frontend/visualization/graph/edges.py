import plotly.graph_objects as go
from typing import List, Set, Tuple, Optional
import numpy as np

from ...data.models import FeatureNode
from ...data.edge_cache import EdgeCache
from ...config.settings import GraphConfig

class EdgeRenderer:
    """Handles rendering of graph edges."""
    
    def __init__(self, config: GraphConfig):
        self.config = config
    
    def create_edge_traces(self, nodes: List[FeatureNode], adjacency_matrix: np.ndarray,
                          selected_feature_id: Optional[int] = None,
                          edge_cache: Optional[EdgeCache] = None) -> List[go.Scatter]:
        """Create edge traces - separate traces for different colors to enable highlighting."""

        # DEBUG: Distinguish regular feature edges from embedding edges
        feature_nodes = [node for node in nodes if node.layer != "embedding" and node.layer != "logit"]
        embedding_nodes = [node for node in nodes if node.layer == "embedding"]

        # Check if any feature edges are going TO embedding nodes (shouldn't happen!)
        if embedding_nodes:
            embedding_ids = {node.id for node in embedding_nodes}

        if edge_cache and edge_cache.is_cached():
            # FAST PATH: Use cached edge data
            return self._create_cached_separate_traces(edge_cache, selected_feature_id)
        else:
            # FALLBACK: Use slow adjacency matrix approach with FILTERING
            return self._create_separate_edge_traces(nodes, adjacency_matrix, selected_feature_id)
    
    def _create_cached_mega_edge_trace(self, edge_cache: EdgeCache, 
                                     selected_feature_id: Optional[int] = None) -> go.Scatter:
        """Create ONE trace containing ALL edges using FAST cached edge data."""
        
        # Get highlighted edges using fast cache lookup
        incoming_keys, outgoing_keys = edge_cache.get_highlighted_edge_sets(selected_feature_id)
        
        # Collect ALL edge coordinates and colors in single arrays
        all_x = []
        all_y = []
        all_colors = []
        all_widths = []
        
        # Get all cached edges (already computed, no matrix traversal needed)
        cached_edges = edge_cache.get_all_edges()
        
        for edge in cached_edges:
            # Add edge coordinates
            all_x.extend([edge.from_node.x, edge.to_node.x, None])
            all_y.extend([edge.from_node.y, edge.to_node.y, None])
            
            # Determine color based on highlighting (fast set lookups)
            edge_key = (edge.from_node_id, edge.to_node_id)
            logit_key = (edge.from_node_id, "logit")
            
            if edge_key in incoming_keys:
                color = self.config.incoming_edge_color
                width = 3
            elif edge_key in outgoing_keys or logit_key in outgoing_keys:
                color = self.config.outgoing_edge_color
                width = 3
            else:
                color = self.config.normal_edge_color
                width = 1.5
            
            # Add color for each segment (3 points per edge)
            all_colors.extend([color, color, color])
            all_widths.extend([width, width, width])
        
        # Create ONE mega trace with all edges - use single color for simplicity  
        return go.Scatter(
            x=all_x,
            y=all_y,
            mode='lines',
            line=dict(width=1, color=self.config.normal_edge_color),
            hoverinfo='none',
            showlegend=False
        )
    
    def _create_single_mega_edge_trace(self, nodes: List[FeatureNode], adjacency_matrix: np.ndarray,
                                      selected_feature_id: Optional[int] = None) -> go.Scatter:
        """Create ONE trace containing ALL edges - much faster than separate traces."""
        
        # Determine highlighted edges
        incoming_edges, outgoing_edges = self._get_highlighted_edges(
            adjacency_matrix, selected_feature_id
        )
        
        # Collect ALL edge coordinates and colors in single arrays
        all_x = []
        all_y = []
        all_colors = []
        all_widths = []
        
        # Create edges between feature nodes
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i < j and i < len(adjacency_matrix) and j < len(adjacency_matrix[0]):
                    weight = adjacency_matrix[node_i.node_id_original, node_j.node_id_original]
                    if weight != 0.0:
                        # Add edge coordinates
                        all_x.extend([node_i.x, node_j.x, None])
                        all_y.extend([node_i.y, node_j.y, None])
                        
                        # Determine color based on highlighting
                        edge_key = (i, j)
                        reverse_key = (j, i)
                        if edge_key in incoming_edges or reverse_key in incoming_edges:
                            color = self.config.incoming_edge_color
                            width = 3
                        elif edge_key in outgoing_edges or reverse_key in outgoing_edges:
                            color = self.config.outgoing_edge_color
                            width = 3
                        else:
                            color = self.config.normal_edge_color
                            width = 1.5
                        
                        # Add color for each segment (3 points per edge)
                        all_colors.extend([color, color, color])
                        all_widths.extend([width, width, width])
        
        # Create edges to logit node
        if len(nodes) > 0:
            logit_node = nodes[-1]
            logit_idx = len(nodes) - 1
            
            if adjacency_matrix.shape[1] > adjacency_matrix.shape[0]:
                logit_col_idx = adjacency_matrix.shape[1] - 1
                
                for i, node_i in enumerate(nodes[:-1]):
                    if (i < len(adjacency_matrix) and 
                        node_i.node_id_original < len(adjacency_matrix) and
                        adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):
                        
                        # Add logit edge coordinates
                        all_x.extend([node_i.x, logit_node.x, None])
                        all_y.extend([node_i.y, logit_node.y, None])
                        
                        # Check if this is a highlighted logit edge
                        logit_key = (i, "logit")
                        if logit_key in outgoing_edges:
                            color = self.config.outgoing_edge_color
                            width = 3
                        else:
                            color = self.config.normal_edge_color
                            width = 1.5
                        
                        all_colors.extend([color, color, color])
                        all_widths.extend([width, width, width])
        
        # Create ONE mega trace with all edges - use single color for simplicity
        return go.Scatter(
            x=all_x,
            y=all_y,
            mode='lines',
            line=dict(width=1, color=self.config.normal_edge_color),
            hoverinfo='none',
            showlegend=False
        )
    
    def _get_highlighted_edges(self, adjacency_matrix: np.ndarray, 
                              selected_feature_id: Optional[int]) -> Tuple[Set, Set]:
        """Get sets of incoming and outgoing edges for selected feature."""
        incoming_edges = set()
        outgoing_edges = set()
        
        if selected_feature_id is not None:
            # Find edges in adjacency matrix
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix[0])):
                    if adjacency_matrix[i, j] != 0:
                        if j == selected_feature_id:
                            incoming_edges.add((i, j))
                        elif i == selected_feature_id:
                            # For edges to logit, j will be the logit column index
                            # but we need to map it to the logit node index in the visualization
                            if j == adjacency_matrix.shape[1] - 1:  # Last column is logit
                                # Map to logit node index (last node in the list)
                                outgoing_edges.add((i, "logit"))  
                            else:
                                outgoing_edges.add((i, j))
        
        return incoming_edges, outgoing_edges
    
    def _create_cached_separate_traces(self, edge_cache: EdgeCache, 
                                     selected_feature_id: Optional[int] = None) -> List[go.Scatter]:
        """Create separate traces by color to enable proper edge highlighting."""
        
        # Get highlighted edges using fast cache lookup
        incoming_keys, outgoing_keys = edge_cache.get_highlighted_edge_sets(selected_feature_id)
        
        # Separate edges by color type
        normal_x, normal_y = [], []
        incoming_x, incoming_y = [], []
        outgoing_x, outgoing_y = [], []
        
        # Get all cached edges
        cached_edges = edge_cache.get_all_edges()
        
        for edge in cached_edges:
            # FILTER: Skip edges involving embedding nodes
            if (hasattr(edge.from_node, 'layer') and edge.from_node.layer == "embedding") or \
               (hasattr(edge.to_node, 'layer') and edge.to_node.layer == "embedding"):
                print(f"🚨 FILTERED CACHED EDGE involving embedding: {edge.from_node.layer} -> {edge.to_node.layer}")
                continue

            edge_key = (edge.from_node_id, edge.to_node_id)
            logit_key = (edge.from_node_id, "logit")

            x_coords = [edge.from_node.x, edge.to_node.x, None]
            y_coords = [edge.from_node.y, edge.to_node.y, None]
            
            if edge_key in incoming_keys:
                incoming_x.extend(x_coords)
                incoming_y.extend(y_coords)
            elif edge_key in outgoing_keys or logit_key in outgoing_keys:
                outgoing_x.extend(x_coords)
                outgoing_y.extend(y_coords)
            else:
                normal_x.extend(x_coords)
                normal_y.extend(y_coords)
        
        traces = []
        
        # Normal edges trace (always add, even if empty)
        traces.append(go.Scatter(
            x=normal_x, y=normal_y, mode='lines',
            line=dict(width=2, color=self.config.normal_edge_color),
            hoverinfo='none', showlegend=False
        ))

        # Incoming edges trace
        if incoming_x:
            traces.append(go.Scatter(
                x=incoming_x, y=incoming_y, mode='lines',
                line=dict(width=5, color=self.config.incoming_edge_color),
                hoverinfo='none', showlegend=False
            ))

        # Outgoing edges trace
        if outgoing_x:
            traces.append(go.Scatter(
                x=outgoing_x, y=outgoing_y, mode='lines',
                line=dict(width=5, color=self.config.outgoing_edge_color),
                hoverinfo='none', showlegend=False
            ))
        
        return traces
    
    def _create_separate_edge_traces(self, nodes: List[FeatureNode], adjacency_matrix: np.ndarray,
                                   selected_feature_id: Optional[int] = None) -> List[go.Scatter]:
        """Create separate traces by color to enable proper edge highlighting."""
        
        # Determine highlighted edges
        incoming_edges, outgoing_edges = self._get_highlighted_edges(
            adjacency_matrix, selected_feature_id
        )
        
        # Separate edges by color type
        normal_x, normal_y = [], []
        incoming_x, incoming_y = [], []
        outgoing_x, outgoing_y = [], []
        
        # Create edges between FEATURE NODES ONLY - exclude embedding and logit nodes
        feature_nodes_only = [node for node in nodes if node.layer != "embedding" and node.layer != "logit"]
        print(f"   ✅ Filtering to {len(feature_nodes_only)} feature nodes only (excluding embedding/logit)")

        for i, node_i in enumerate(feature_nodes_only):
            for j, node_j in enumerate(feature_nodes_only):
                if i < j and i < len(adjacency_matrix) and j < len(adjacency_matrix[0]):
                    weight = adjacency_matrix[node_i.node_id_original, node_j.node_id_original]
                    if weight != 0.0:
                        x_coords = [node_i.x, node_j.x, None]
                        y_coords = [node_i.y, node_j.y, None]
                        
                        edge_key = (i, j)
                        reverse_key = (j, i)
                        if edge_key in incoming_edges or reverse_key in incoming_edges:
                            incoming_x.extend(x_coords)
                            incoming_y.extend(y_coords)
                        elif edge_key in outgoing_edges or reverse_key in outgoing_edges:
                            outgoing_x.extend(x_coords)
                            outgoing_y.extend(y_coords)
                        else:
                            normal_x.extend(x_coords)
                            normal_y.extend(y_coords)
        
        # Create edges to logit node from FEATURE NODES ONLY
        logit_nodes = [node for node in nodes if node.layer == "logit"]
        logit_edges_created = 0

        if logit_nodes:
            logit_node = logit_nodes[0]  # Should only be one logit node
            print(f"   🎯 LOGIT NODE found at ({logit_node.x:.1f}, {logit_node.y:.1f}) - ID: {logit_node.id}")

            if adjacency_matrix.shape[1] > adjacency_matrix.shape[0]:
                logit_col_idx = adjacency_matrix.shape[1] - 1
                print(f"   🎯 Checking logit column {logit_col_idx} for edges to logit")

                for i, node_i in enumerate(feature_nodes_only):
                    if (i < len(adjacency_matrix) and
                        node_i.node_id_original < len(adjacency_matrix) and
                        adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):

                        logit_edges_created += 1
                        print(f"   ✅ Creating logit edge: Feature {node_i.id} -> Logit")
                        
                        x_coords = [node_i.x, logit_node.x, None]
                        y_coords = [node_i.y, logit_node.y, None]
                        
                        logit_key = (i, "logit")
                        if logit_key in outgoing_edges:
                            outgoing_x.extend(x_coords)
                            outgoing_y.extend(y_coords)
                        else:
                            normal_x.extend(x_coords)
                            normal_y.extend(y_coords)
        else:
            print("   ❌ NO LOGIT NODE FOUND!")

        print(f"   📊 LOGIT EDGES SUMMARY: {logit_edges_created} edges created to logit")
        if logit_edges_created == 0:
            print("   ⚠️  WARNING: No logit edges! Expected feature->logit connections missing!")

        traces = []

        # Normal edges trace (always add, even if empty)
        traces.append(go.Scatter(
            x=normal_x, y=normal_y, mode='lines',
            line=dict(width=2, color=self.config.normal_edge_color),
            hoverinfo='none', showlegend=False
        ))

        # Incoming edges trace
        if incoming_x:
            traces.append(go.Scatter(
                x=incoming_x, y=incoming_y, mode='lines',
                line=dict(width=5, color=self.config.incoming_edge_color),
                hoverinfo='none', showlegend=False
            ))

        # Outgoing edges trace
        if outgoing_x:
            traces.append(go.Scatter(
                x=outgoing_x, y=outgoing_y, mode='lines',
                line=dict(width=5, color=self.config.outgoing_edge_color),
                hoverinfo='none', showlegend=False
            ))
        
        return traces
    
    def _create_single_edge(self, node_from: FeatureNode, node_to: FeatureNode,
                           weight: float, from_idx: int, to_idx: int,
                           incoming_edges: Set, outgoing_edges: Set) -> go.Scatter:
        """Create a single edge trace."""
        edge_key = (from_idx, to_idx)
        
        # Handle special case for edges to logit node
        logit_edge_key = (from_idx, "logit")
        
        # Determine edge color and width
        if edge_key in incoming_edges:
            edge_color = self.config.incoming_edge_color
            edge_width = 3
        elif edge_key in outgoing_edges or logit_edge_key in outgoing_edges:
            edge_color = self.config.outgoing_edge_color
            edge_width = 3
        else:
            edge_color = self.config.normal_edge_color
            edge_width = 1.5
        
        return go.Scatter(
            x=[node_from.x, node_to.x, None],
            y=[node_from.y, node_to.y, None],
            mode='lines',
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='none',
            showlegend=False
        )
