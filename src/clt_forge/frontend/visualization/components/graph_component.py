from dash import dcc, html
from typing import Optional, Set

from ...config.settings import AppConfig
from ...data.loaders import DataLoader
from ...data.models import GraphData, FeatureEdge
from ...data.edge_cache import EdgeCache
from ..graph.layout import GraphLayoutCalculator
from ..graph.renderer import GraphRenderer

class GraphComponent:
    """Main graph component that orchestrates graph creation."""
    
    # Class-level flag to prevent duplicate precomputation across all instances
    _global_precomputation_complete = False
    
    def __init__(self, config: AppConfig, data_loader: DataLoader, layout_calculator: GraphLayoutCalculator):
        self.config = config
        self.data_loader = data_loader
        self.layout_calculator = layout_calculator
        self.graph_renderer = GraphRenderer(config.graph)
        self._graph_data = None
        self._static_figure = None  # Cache static figure for fast updates
        self._cluster_static_figure = None  # Cache cluster figure for fast updates
        self._last_highlighted = None  # Track last highlighted node and connections
        self._original_colors = None  # Cache original node colors
        self._cluster_original_colors = None  # Cache cluster node colors
        
        # PERFORMANCE: Edge cache for fast edge lookups
        self.edge_cache = EdgeCache()
        
        # PERFORMANCE: Pre-computed figures for instant highlighting
        self._precomputed_figures = {}  # {node_id: figure, 'base': base_figure}
        self._precomputation_complete = False
        self._precomputation_progress = 0  # Progress percentage (0-100)
        
        # Set default screen dimensions for graph area (68% of screen)
        self.screen_width = 1088  # 68% of 1600
        self.screen_height = 900
    
    def set_screen_dimensions(self, width: int, height: int):
        """Set screen dimensions for responsive layout."""
        # Use 68% of width to match the actual layout allocation in app.py
        new_screen_width = int(width * 0.68)
        new_screen_height = height - 96  # Account for header
        
        # DEBUG: Log all calls to set_screen_dimensions
        print(f"🔍 set_screen_dimensions called: {width}x{height} → {new_screen_width}x{new_screen_height}")
        print(f"🔍 Current dimensions: {self.screen_width}x{self.screen_height}")
        # Precomputation will be triggered on first graph request
        
        # PERFORMANCE: Only clear cache if dimensions changed significantly
        width_changed = abs(new_screen_width - self.screen_width) > 50
        height_changed = abs(new_screen_height - self.screen_height) > 50
        
        print(f"🔍 Width changed: {width_changed} (diff: {abs(new_screen_width - self.screen_width)})")
        print(f"🔍 Height changed: {height_changed} (diff: {abs(new_screen_height - self.screen_height)})")
        
        if width_changed or height_changed:
            print(f"📐 Screen dimensions changed significantly: {self.screen_width}x{self.screen_height} → {new_screen_width}x{new_screen_height}")
            
            # Update dimensions
            self.screen_width = new_screen_width
            self.screen_height = new_screen_height
            
            self.layout_calculator.set_screen_dimensions(self.screen_width, self.screen_height)
            self.graph_renderer.set_screen_dimensions(self.screen_width, self.screen_height)
            
            # Clear cache to recalculate with new dimensions
            self._graph_data = None
            # Clear pre-computed figures - they need to be regenerated for new dimensions
            self._precomputed_figures.clear()
            self._precomputation_complete = False
            self._precomputation_progress = 0
            # Don't reset global flag - let it stay true to prevent unnecessary recomputation
            print("🧹 Cache cleared due to screen dimension change")
        else:
            # Small changes - just update dimensions without clearing cache
            print("✅ Dimensions unchanged, keeping cache")
            self.screen_width = new_screen_width
            self.screen_height = new_screen_height
    
    def get_graph_data(self) -> GraphData:
        """Get or create graph data with current screen dimensions."""
        if self._graph_data is None:
            # Force clear data loader cache to get fresh intersection features
            self.data_loader.clear_cache()
            base_data = self.data_loader.preprocess_data()
            nodes = self.layout_calculator.calculate_node_positions(base_data)
            edges = self._create_edges(base_data, nodes)
            
            # Get processed intervention data (already filtered for active features)
            intervention_data = self.data_loader.get_processed_intervention_data()
            
            self._graph_data = GraphData(
                nodes=nodes,
                edges=edges,
                adjacency_matrix=base_data.adjacency_matrix,
                active_mask=base_data.active_mask,
                feature_indices=base_data.feature_indices,
                input_tokens=base_data.input_tokens,
                input_str=base_data.input_str,
                n_layers=base_data.n_layers,
                prompt_length=base_data.prompt_length,
                token_x_positions=base_data.token_x_positions,
                top_logit_token=base_data.top_logit_token,
                top5_logit_tokens=getattr(base_data, 'top5_logit_tokens', None),
                top5_logit_probs=getattr(base_data, 'top5_logit_probs', None),
                intervention_data=intervention_data,
                feature_list_intersection=getattr(base_data, 'feature_list_intersection', None),
                embedding_adjacency=getattr(base_data, 'embedding_adjacency', None)
            )
            
            
            # Build edge cache for fast edge lookups
            self.edge_cache.build_cache(nodes, base_data.adjacency_matrix)
        
        return self._graph_data
    
    def _precompute_all_figures(self):
        """Pre-compute all possible node highlighting states for instant clicking."""
        # Use class-level flag to prevent any duplicate precomputation
        if GraphComponent._global_precomputation_complete or self._precomputation_complete:
            return
        
        # Add instance-level lock to prevent concurrent execution within same instance
        if hasattr(self, '_precomputing_in_progress') and self._precomputing_in_progress:
            return
            
        # Set both global and instance flags immediately
        GraphComponent._global_precomputation_complete = True
        self._precomputing_in_progress = True
            
        import time
        start_time = time.time()
        
        graph_data = self.get_graph_data()
        num_nodes = len(graph_data.nodes)
        
        print(f"Precomputing {num_nodes + 1} graph figures...")
        
        # Use the original working approach but with better optimization
        self._precomputed_figures = self._create_working_precomputed_figures(graph_data)
        
        total_time = time.time() - start_time
        self._precomputation_complete = True
        self._precomputation_progress = 100
        self._precomputing_in_progress = False  # Clear the lock
        print(f"Precomputation complete: {total_time:.1f}s for {len(self._precomputed_figures)} figures")
    
    def _hyper_fast_bulk_create(self, graph_data):
        """Create all figures with absolute minimal computation."""
        import plotly.graph_objects as go
        import time
        
        num_nodes = len(graph_data.nodes)
        figures = {}
        
        print(f"    ⚡ Creating template data once for {num_nodes} nodes...")
        
        # Pre-compute coordinates and hover text ONCE
        x_coords = [node.x for node in graph_data.nodes]
        y_coords = [node.y for node in graph_data.nodes]
        hover_texts = [f"Layer {node.layer}, Pos {node.pos}, Feat {node.feature_idx}" for node in graph_data.nodes]
        
        # Create layout ONCE
        layout = self._create_optimized_layout()
        
        # Base colors
        base_colors = ['#f8f9fa'] * num_nodes
        base_sizes = [10] * num_nodes
        
        print("    ⚡ Creating base figure with NO EDGES (maximum speed)...")
        # Base figure - nodes only, NO EDGES for maximum speed
        base_node_trace = go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers',
            marker=dict(size=base_sizes, color=base_colors, line=dict(width=2, color='black')),
            text=hover_texts, hoverinfo='text', showlegend=False
        )
        figures['base'] = go.Figure(data=[base_node_trace], layout=layout)
        
        print(f"    ⚡ Batch creating {num_nodes} node highlighting variants...")
        create_start = time.time()
        
        # Create all variants by just changing node colors (NO edge computation)
        for node_id in range(num_nodes):
            if node_id % 100 == 0:
                elapsed = time.time() - create_start  
                print(f"      📈 {node_id}/{num_nodes} ({elapsed:.1f}s)")
            
            # ULTRA-FAST: Just create new marker colors, reuse everything else
            colors = base_colors.copy()
            sizes = base_sizes.copy()
            
            # Selected node - red
            colors[node_id] = '#E74C3C'
            sizes[node_id] = 16
            
            # Create node trace with highlighting
            highlighted_trace = go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers', 
                marker=dict(size=sizes, color=colors, line=dict(width=2, color='black')),
                text=hover_texts, hoverinfo='text', showlegend=False
            )
            
            # NO EDGE COMPUTATION - just nodes
            figures[node_id] = go.Figure(data=[highlighted_trace], layout=layout)
        
        create_time = time.time() - create_start
        print(f"    ✅ All {num_nodes} variants created in {create_time:.1f}s")
        print(f"    📊 Speed: {create_time/num_nodes*1000:.1f}ms per figure")
        
        return figures
    
    def _create_working_precomputed_figures(self, graph_data):
        """Create ALL figures in dictionary using the most efficient approach possible."""
        # Create all figures efficiently
        
        figures = {}
        num_nodes = len(graph_data.nodes)
        
        # Create base figure using efficient renderer (with intersection features)
        base_figure = self.graph_renderer.create_figure(
            graph_data, 
            selected_feature_id=None, 
            nodes_with_descriptions=set(),
            node_to_cluster={},
            cluster_highlighted_nodes=set(),
            edge_cache=self.edge_cache
        )
        figures['base'] = base_figure
        
        # Create all highlighted variants using the renderer directly
        for node_id in range(num_nodes):
            # Update progress for every 10% completion
            if node_id % max(1, num_nodes // 10) == 0:
                progress = int((node_id / num_nodes) * 100)
                self._precomputation_progress = progress
                print(f"Progress: {progress}% ({node_id}/{num_nodes})")
            
            # Use the renderer directly with optimized edge system
            highlighted_figure = self.graph_renderer.create_figure(
                graph_data,
                selected_feature_id=node_id,
                nodes_with_descriptions=set(),
                node_to_cluster={},
                cluster_highlighted_nodes=set(),
                edge_cache=self.edge_cache
            )
            figures[node_id] = highlighted_figure
        
        # All figures precomputed successfully
        return figures
    
    def get_precomputation_progress(self):
        """Get current precomputation progress percentage (0-100)."""
        return self._precomputation_progress
    
    def _bulk_create_figures(self, graph_data, all_highlighted_sets, all_edge_sets):
        """Create all figure variants in one optimized pass."""
        import plotly.graph_objects as go
        import time
        
        figures = {}
        num_nodes = len(graph_data.nodes)
        
        print(f"    📐 Pre-computing common data for {num_nodes} nodes...")
        # Pre-compute common data once
        coords_start = time.time()
        x_coords = [node.x for node in graph_data.nodes]
        y_coords = [node.y for node in graph_data.nodes]
        hover_texts = [f"Layer {node.layer}, Pos {node.pos}, Feat {node.feature_idx}" for node in graph_data.nodes]
        
        # Base colors and sizes
        base_colors = ['#f8f9fa'] * len(graph_data.nodes)
        base_sizes = [10] * len(graph_data.nodes)
        coords_time = time.time() - coords_start
        print(f"    ✅ Common data ready ({coords_time:.3f}s)")
        
        # OPTIMIZATION: Skip expensive edge trace creation for now - use shared edges
        print("    🔗 Creating shared edge traces (ultra-fast approach)...")
        edges_start = time.time()
        shared_edges = self._create_shared_edge_traces(graph_data)
        edges_time = time.time() - edges_start
        print(f"    ✅ Shared edge traces ready ({edges_time:.1f}s)")
        
        # Create base figure
        print("    🎨 Creating base node trace...")
        base_start = time.time()
        node_trace = go.Scatter(
            x=x_coords, y=y_coords,
            mode='markers',
            marker=dict(
                size=base_sizes,
                color=base_colors,
                line=dict(width=2, color='black')
            ),
            text=hover_texts,
            hoverinfo='text',
            showlegend=False
        )
        
        # Base figure
        base_edge_count = len(shared_edges)
        figures['base'] = go.Figure(
            data=[node_trace] + shared_edges,
            layout=self._create_optimized_layout()
        )
        base_time = time.time() - base_start
        print(f"    ✅ Base figure created with {base_edge_count} edges ({base_time:.3f}s)")
        
        # Create highlighted variants by copying and modifying node trace ONLY (reuse edges)
        print(f"    🌈 Creating {num_nodes} highlighted node variants (nodes only, shared edges)...")
        variants_start = time.time()
        
        for idx, (node_id, highlighted_nodes) in enumerate(all_highlighted_sets.items()):
            if node_id == 'base':
                continue
            
            if idx % 100 == 0:
                elapsed = time.time() - variants_start
                print(f"      📈 Variants progress: {idx}/{num_nodes} ({elapsed:.1f}s)")
                
            # Create new node trace with highlighting colors (FAST)
            colors = base_colors.copy()
            sizes = base_sizes.copy()
            
            # Selected node
            colors[node_id] = '#E74C3C'  # Red
            sizes[node_id] = 16
            
            # Highlighted connected nodes
            highlight_count = 0
            for h_node in highlighted_nodes:
                if h_node != node_id and h_node < len(colors):
                    colors[h_node] = '#F39C12'  # Orange
                    sizes[h_node] = 12
                    highlight_count += 1
            
            # Create new node trace (much faster than copying)
            highlighted_node_trace = go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=2, color='black')
                ),
                text=hover_texts,
                hoverinfo='text',
                showlegend=False
            )
            
            # ULTRA-FAST: Reuse same edge traces for all figures (no edge highlighting for now)
            figures[node_id] = go.Figure(
                data=[highlighted_node_trace] + shared_edges,
                layout=self._create_optimized_layout()
            )
            
            if idx < 3:  # Debug first few
                print(f"      🔍 Node {node_id}: {highlight_count} highlighted nodes, {base_edge_count} shared edges")
        
        variants_time = time.time() - variants_start
        print(f"    ✅ All variants created with shared edges ({variants_time:.1f}s)")
        
        return figures
    
    def _create_shared_edge_traces(self, graph_data):
        """Create one set of edge traces to be shared across all figures (ultra-fast)."""
        import plotly.graph_objects as go
        
        print("      🔍 Creating single set of shared edges...")
        shared_edges = []
        
        # Get all edges that exist in the graph
        existing_edges = []
        
        # Scan for feature-to-feature edges
        feature_edges = 0
        for i, node_i in enumerate(graph_data.nodes):
            for j, node_j in enumerate(graph_data.nodes):
                if i < j and (i < len(graph_data.adjacency_matrix) and 
                             j < len(graph_data.adjacency_matrix[0])):
                    weight = graph_data.adjacency_matrix[node_i.node_id_original, node_j.node_id_original]
                    if weight != 0.0:
                        existing_edges.append((i, j, node_i, node_j, weight))
                        feature_edges += 1
        
        # Add edges to logit node
        logit_edges = 0
        if len(graph_data.nodes) > 0:
            logit_node = graph_data.nodes[-1]
            logit_col_idx = graph_data.adjacency_matrix.shape[1] - 1
            for i, node_i in enumerate(graph_data.nodes[:-1]):
                if (node_i.node_id_original < len(graph_data.adjacency_matrix) and
                    graph_data.adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):
                    weight = graph_data.adjacency_matrix[node_i.node_id_original, logit_col_idx]
                    existing_edges.append((i, len(graph_data.nodes)-1, node_i, logit_node, weight))
                    logit_edges += 1
        
        print(f"      ✅ Found {len(existing_edges)} edges ({feature_edges} feature-feature, {logit_edges} to-logit)")
        
        # Create shared edge traces (all gray for now - no highlighting)
        for i, j, node_i, node_j, weight in existing_edges:
            edge_trace = go.Scatter(
                x=[node_i.x, node_j.x, None],
                y=[node_i.y, node_j.y, None],
                mode='lines',
                line=dict(width=1, color='rgba(0, 0, 0, 0.05)'),
                hoverinfo='none',
                showlegend=False
            )
            shared_edges.append(edge_trace)
        
        print(f"      ✅ Created {len(shared_edges)} shared edge traces")
        return shared_edges
    
    def _bulk_create_edge_traces(self, graph_data, all_edge_sets):
        """Create all edge trace variants in one optimized pass."""
        import plotly.graph_objects as go
        import time
        
        all_edge_traces = {}
        edges_start = time.time()
        
        print("      🔍 Scanning adjacency matrix for existing edges...")
        # Get all edges that exist in the graph
        existing_edges = []
        adj_scan_start = time.time()
        
        num_nodes = len(graph_data.nodes)
        adj_shape = graph_data.adjacency_matrix.shape
        
        # Scan for feature-to-feature edges
        feature_edges = 0
        for i, node_i in enumerate(graph_data.nodes):
            for j, node_j in enumerate(graph_data.nodes):
                if i < j and (i < len(graph_data.adjacency_matrix) and 
                             j < len(graph_data.adjacency_matrix[0])):
                    weight = graph_data.adjacency_matrix[node_i.node_id_original, node_j.node_id_original]
                    if weight != 0.0:
                        existing_edges.append((i, j, node_i, node_j, weight))
                        feature_edges += 1
        
        # Add edges to logit node
        logit_edges = 0
        if len(graph_data.nodes) > 0:
            logit_node = graph_data.nodes[-1]
            logit_col_idx = graph_data.adjacency_matrix.shape[1] - 1
            for i, node_i in enumerate(graph_data.nodes[:-1]):
                if (node_i.node_id_original < len(graph_data.adjacency_matrix) and
                    graph_data.adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):
                    weight = graph_data.adjacency_matrix[node_i.node_id_original, logit_col_idx]
                    existing_edges.append((i, len(graph_data.nodes)-1, node_i, logit_node, weight))
                    logit_edges += 1
        
        adj_scan_time = time.time() - adj_scan_start
        total_edges = len(existing_edges)
        print(f"      ✅ Found {total_edges} edges ({feature_edges} feature-feature, {logit_edges} to-logit) in {adj_scan_time:.1f}s")
        
        # Create base edge traces (no highlighting)
        print("      🎨 Creating base edge traces...")
        base_start = time.time()
        base_edges = []
        for i, j, node_i, node_j, weight in existing_edges:
            edge_trace = go.Scatter(
                x=[node_i.x, node_j.x, None],
                y=[node_i.y, node_j.y, None],
                mode='lines',
                line=dict(width=1, color='rgba(0, 0, 0, 0.05)'),
                hoverinfo='none',
                showlegend=False
            )
            base_edges.append(edge_trace)
        
        all_edge_traces['base'] = base_edges
        base_time = time.time() - base_start
        print(f"      ✅ Base edges created ({base_time:.3f}s)")
        
        # Create highlighted edge traces for each node by COPYING base edges and modifying colors
        print(f"      🌈 Creating highlighted edge variants for {num_nodes} nodes by copying base edges...")
        variants_start = time.time()
        
        processed_nodes = 0
        for node_id, (incoming_edges, outgoing_edges) in all_edge_sets.items():
            if node_id == 'base':
                continue
                
            if processed_nodes % 50 == 0:
                elapsed = time.time() - variants_start
                print(f"        📈 Edge variants: {processed_nodes}/{num_nodes} ({elapsed:.1f}s)")
            
            # OPTIMIZATION: Copy base edges and modify colors instead of recreating
            highlighted_edges = []
            incoming_count = 0
            outgoing_count = 0
            normal_count = 0
            
            for edge_idx, (i, j, node_i, node_j, weight) in enumerate(existing_edges):
                # Copy the base edge trace
                base_edge = base_edges[edge_idx]
                edge_trace_copy = go.Scatter(base_edge)
                
                edge_key = (i, j)
                reverse_key = (j, i)
                logit_key = (i, "logit")
                
                # Determine edge color and modify the copy
                if (edge_key in incoming_edges or reverse_key in incoming_edges):
                    edge_trace_copy.line.color = 'rgba(34, 197, 94, 0.9)'  # Green for incoming
                    edge_trace_copy.line.width = 2
                    incoming_count += 1
                elif (edge_key in outgoing_edges or reverse_key in outgoing_edges or logit_key in outgoing_edges):
                    edge_trace_copy.line.color = 'rgba(59, 130, 246, 0.9)'  # Blue for outgoing
                    edge_trace_copy.line.width = 2
                    outgoing_count += 1
                else:
                    # Keep base edge color and width (already set)
                    normal_count += 1
                
                highlighted_edges.append(edge_trace_copy)
            
            all_edge_traces[node_id] = highlighted_edges
            processed_nodes += 1
            
            if processed_nodes <= 3:  # Debug first few
                print(f"        🔍 Node {node_id} edges: {incoming_count} incoming, {outgoing_count} outgoing, {normal_count} normal")
        
        variants_time = time.time() - variants_start
        total_time = time.time() - edges_start
        print(f"      ✅ All edge variants created by copying in ({variants_time:.1f}s)")
        print(f"      📊 Total edge processing: {total_time:.1f}s for {len(all_edge_traces)} variants")
        
        return all_edge_traces
    
    def _create_optimized_layout(self):
        """Create optimized layout for pre-computed figures."""
        return dict(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=40),
            annotations=[],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=self.screen_width,
            height=self.screen_height,
            plot_bgcolor=self.config.graph.background_color,
            paper_bgcolor=self.config.graph.background_color
        )
    
    def create_graph_figure(self, selected_feature_id: Optional[int] = None, 
                           nodes_with_descriptions: Set[int] = None, 
                           cluster_data: dict = None,
                           cluster_highlighted_nodes: Set[int] = None):
        """Create the graph figure with responsive dimensions and cluster coloring."""
        
        # INSTANT PATH: Use pre-computed figures when no clusters involved
        if ((cluster_data is None or cluster_data == False or cluster_data == {}) and 
            (cluster_highlighted_nodes is None or len(cluster_highlighted_nodes) == 0) and
            (nodes_with_descriptions is None or len(nodes_with_descriptions) == 0)):
            
            # Trigger pre-computation if not done yet
            self._precompute_all_figures()
            
            # INSTANT RESPONSE: Just return pre-computed figure
            # Return pre-computed figures when available
            
            if selected_feature_id is None:
                if 'base' in self._precomputed_figures:
                    return self._precomputed_figures['base']
            elif selected_feature_id in self._precomputed_figures:
                return self._precomputed_figures[selected_feature_id]
            
            # Fallback: figures not available, trigger precomputation
            # Reset global flag to allow recomputation if cache was cleared
            if not self._precomputed_figures:
                GraphComponent._global_precomputation_complete = False
                self._precompute_all_figures()
                # Retry after precomputation
                if selected_feature_id is None and 'base' in self._precomputed_figures:
                    return self._precomputed_figures['base']
                elif selected_feature_id in self._precomputed_figures:
                    return self._precomputed_figures[selected_feature_id]
        
        # CLUSTER PATH: Handle cluster highlighting (less common, can be slower)
        graph_data = self.get_graph_data()
        
        if cluster_data and self._cluster_static_figure is not None:
            # Use cluster-aware fast highlighting when clusters are active
            return self._create_ultra_fast_highlight_with_clusters(selected_feature_id, graph_data, cluster_highlighted_nodes)
        
        # Full recreation for complex cases
        # Slow path: full recreation (only for first load or complex changes)
        # Create node to cluster mapping
        node_to_cluster = {}
        if cluster_data:
            for cluster_id, cluster_info in cluster_data.items():
                cluster_color = cluster_info.get('color', '#3B82F6')
                for node_data in cluster_info.get('nodes', []):
                    node_index = node_data.get('node_index')
                    if node_index is not None:
                        node_to_cluster[node_index] = cluster_color
        
        figure = self.graph_renderer.create_figure(
            graph_data, 
            selected_feature_id, 
            nodes_with_descriptions or set(),
            node_to_cluster,
            cluster_highlighted_nodes or set(),
            edge_cache=self.edge_cache
        )
        
        # Ensure consistent styling: black borders and full opacity
        if figure.data and len(figure.data) > 0 and hasattr(figure.data[0], 'marker'):
            # Always ensure black borders and full opacity
            figure.data[0].marker.line = dict(width=2, color='black')
            figure.data[0].marker.opacity = 1.0
        # Cache static version for future fast updates
        if not cluster_data and not cluster_highlighted_nodes:
            self._static_figure = figure
            # Cache original colors for proper restoration
            if figure.data and len(figure.data) > 0 and hasattr(figure.data[0], 'marker'):
                marker_color = figure.data[0].marker.color
                if isinstance(marker_color, str):
                    # Single color for all nodes - expand to list based on number of nodes
                    num_nodes = len(figure.data[0].x) if hasattr(figure.data[0], 'x') else 0
                    self._original_colors = [marker_color] * num_nodes
                elif marker_color is not None:
                    # Already a list of colors
                    self._original_colors = list(marker_color)
                else:
                    # No color set, use default
                    num_nodes = len(figure.data[0].x) if hasattr(figure.data[0], 'x') else 0
                    self._original_colors = ['#f8f9fa'] * num_nodes
        elif cluster_data:
            # Cache cluster version for future fast updates with clusters
            self._cluster_static_figure = figure
            if figure.data and len(figure.data) > 0 and hasattr(figure.data[0], 'marker'):
                marker_color = figure.data[0].marker.color
                if isinstance(marker_color, str):
                    # Single color for all nodes - expand to list based on number of nodes
                    num_nodes = len(figure.data[0].x) if hasattr(figure.data[0], 'x') else 0
                    self._cluster_original_colors = [marker_color] * num_nodes
                elif marker_color is not None:
                    # Already a list of colors
                    self._cluster_original_colors = list(marker_color)
                else:
                    # No color set, use default
                    num_nodes = len(figure.data[0].x) if hasattr(figure.data[0], 'x') else 0
                    self._cluster_original_colors = ['#f8f9fa'] * num_nodes
        
        return figure
    
    def _create_ultra_fast_highlight(self, selected_feature_id, graph_data):
        """Ultra-fast highlighting that only touches specific nodes."""
        import copy
        
        # PERFORMANCE: Create new figure with copied traces to avoid Plotly constraints
        import plotly.graph_objects as go
        figure = go.Figure(data=[copy.copy(trace) for trace in self._static_figure.data], 
                          layout=copy.copy(self._static_figure.layout))
        
        # COLORS - Keep original defaults, enhance highlighting  
        selected_color = '#E74C3C'              # Vibrant red
        connected_color = '#F39C12'             # Vibrant orange
        
        if selected_feature_id is None:
            # CLEAR HIGHLIGHTING - Return new figure with original data and proper z-order
            copied_traces = [copy.copy(trace) for trace in self._static_figure.data]
            # Move node trace to end so it appears above edges
            if len(copied_traces) > 0:
                node_trace = copied_traces[0]
                other_traces = copied_traces[1:]
                copied_traces = other_traces + [node_trace]
            return go.Figure(data=copied_traces, layout=copy.copy(self._static_figure.layout))
            
        else:
            # PERFORMANCE: Get the node trace (first trace is already copied)
            node_trace = figure.data[0]
            
            # Start with original colors (fast array copy)
            if self._original_colors:
                colors = self._original_colors[:]
            else:
                marker_color = node_trace.marker.color
                if isinstance(marker_color, str):
                    # Single color for all nodes - expand to list
                    num_nodes = len(node_trace.x) if hasattr(node_trace, 'x') else 0
                    colors = [marker_color] * num_nodes
                else:
                    colors = list(marker_color)
            
            # PERFORMANCE: Pre-calculate nodes to avoid repeated calls
            highlighted_nodes = self._get_highlighted_nodes(graph_data, selected_feature_id)
            
            # CLEAR PREVIOUS HIGHLIGHTING first (if any) - optimized loop
            if self._last_highlighted is not None and self._original_colors is not None:
                # Batch restore previously highlighted nodes
                for node_id in self._last_highlighted['nodes']:
                    if node_id < len(colors):
                        colors[node_id] = self._original_colors[node_id]
            
            # HIGHLIGHT NEW NODES - optimized assignments with bounds checking
            if selected_feature_id < len(colors):
                colors[selected_feature_id] = selected_color
            for node_id in highlighted_nodes:
                if node_id != selected_feature_id and node_id < len(colors):
                    colors[node_id] = connected_color
            
            # ENHANCED NODE SIZES - Make highlighted nodes bigger
            sizes = list(node_trace.marker.size) if hasattr(node_trace.marker, 'size') and node_trace.marker.size else [10] * len(colors)
            if selected_feature_id < len(sizes):
                sizes[selected_feature_id] = 14  # Bigger selected node (same as selected_node_size)
            for node_id in highlighted_nodes:
                if node_id != selected_feature_id and node_id < len(sizes):
                    sizes[node_id] = 10  # Same size as normal nodes
            
            # Apply updates in batch - Keep highlighting colors but ALWAYS use black borders
            node_trace.marker.color = colors  # Keep the highlighting colors
            node_trace.marker.size = sizes
            node_trace.marker.line = dict(width=2, color='black')  # FORCE black borders
            
            # PERFORMANCE: Only update edges if they changed
            incoming_edges, outgoing_edges = self._get_highlighted_edges(graph_data, selected_feature_id)
            current_edges = incoming_edges | outgoing_edges
            
            # Only update edges if different from last time
            last_edges = self._last_highlighted['edges'] if self._last_highlighted else set()
            if current_edges != last_edges:
                self._update_figure_edges(figure, incoming_edges, outgoing_edges, graph_data)
            
            # TRACK current highlighting for next time
            all_highlighted_nodes = highlighted_nodes | {selected_feature_id}
            self._last_highlighted = {
                'nodes': all_highlighted_nodes,
                'edges': current_edges
            }
            
            # PERFORMANCE: Reorder traces so nodes appear above edges
            # Move node trace to the end so it renders on top
            node_trace = figure.data[0]
            other_traces = list(figure.data[1:])
            figure = go.Figure(data=other_traces + [node_trace], layout=figure.layout)
        
        return figure
    
    def _create_ultra_fast_highlight_with_clusters(self, selected_feature_id, graph_data, cluster_highlighted_nodes):
        """Ultra-fast highlighting that preserves cluster colors."""
        import copy
        import plotly.graph_objects as go
        
        print(f"Ultra-fast cluster highlighting: cluster_highlighted_nodes={list(cluster_highlighted_nodes) if cluster_highlighted_nodes else []}")
        
        # PERFORMANCE: Create new figure with copied traces from cluster static figure
        figure = go.Figure(data=[copy.copy(trace) for trace in self._cluster_static_figure.data], 
                          layout=copy.copy(self._cluster_static_figure.layout))
        
        # Get the node trace (first trace is already copied)
        node_trace = figure.data[0]
        
        # Start with cluster colors (fast array copy)
        if self._cluster_original_colors:
            colors = self._cluster_original_colors[:]
        else:
            marker_color = node_trace.marker.color
            if isinstance(marker_color, str):
                # Single color for all nodes - expand to list
                num_nodes = len(node_trace.x) if hasattr(node_trace, 'x') else 0
                colors = [marker_color] * num_nodes
            else:
                colors = list(marker_color)
        sizes = list(node_trace.marker.size) if hasattr(node_trace.marker, 'size') and node_trace.marker.size else [12] * len(colors)
        
        # CLUSTER HIGHLIGHTING: Make cluster nodes bigger and more visible
        if cluster_highlighted_nodes:
            cluster_highlight_color = '#F59E0B'  # Orange/yellow for cluster highlighting
            for node_id in cluster_highlighted_nodes:
                if node_id < len(colors):
                    colors[node_id] = cluster_highlight_color  # Bright orange for cluster nodes
                    sizes[node_id] = 14  # Bigger size for cluster nodes
                    print(f"HIGHLIGHTED CLUSTER NODE {node_id}: color={cluster_highlight_color}, size=16")
        
        # NODE HIGHLIGHTING: Only if individual node is selected AND no cluster highlighting
        if selected_feature_id is not None and not cluster_highlighted_nodes:
            selected_color = '#E74C3C'              # Vibrant red
            connected_color = '#F39C12'             # Vibrant orange
            
            # PERFORMANCE: Pre-calculate nodes to avoid repeated calls
            highlighted_nodes = self._get_highlighted_nodes(graph_data, selected_feature_id)
            
            # HIGHLIGHT NEW NODES - overlay on cluster colors
            colors[selected_feature_id] = selected_color
            sizes[selected_feature_id] = 14
            for node_id in highlighted_nodes:
                if node_id != selected_feature_id and node_id < len(colors):
                    colors[node_id] = connected_color
                    sizes[node_id] = 10
            
            # Handle edges
            incoming_edges, outgoing_edges = self._get_highlighted_edges(graph_data, selected_feature_id)
            self._update_figure_edges(figure, incoming_edges, outgoing_edges, graph_data)
        
        # Apply updates in batch - Keep highlighting colors but ALWAYS use black borders
        node_trace.marker.color = colors  # Keep the highlighting colors
        node_trace.marker.size = sizes
        node_trace.marker.line = dict(width=2, color='black')  # FORCE black borders
        
        # Reorder traces so nodes appear above edges
        node_trace = figure.data[0]
        other_traces = list(figure.data[1:])
        figure = go.Figure(data=other_traces + [node_trace], layout=figure.layout)
        
        return figure
    
    def _update_figure_edges(self, figure, incoming_edges, outgoing_edges, graph_data):
        """Update edge colors directly in the figure."""
        # ENHANCED EDGE COLORS - Thinner for cleaner look
        incoming_edge_color = 'rgba(34, 197, 94, 0.9)'    # Bright green
        outgoing_edge_color = 'rgba(59, 130, 246, 0.9)'   # Bright blue
        default_edge_color = 'rgba(0, 0, 0, 0.05)'        # Original subtle gray
        
        edge_map = self._create_edge_mapping(graph_data)
        all_highlighted_edges = incoming_edges | outgoing_edges
        
        # Reset previously highlighted edges first
        if self._last_highlighted is not None:
            for edge_key in self._last_highlighted['edges']:
                if edge_key in edge_map and edge_map[edge_key] < len(figure.data):
                    trace_idx = edge_map[edge_key]
                    figure.data[trace_idx].line.color = default_edge_color
                    figure.data[trace_idx].line.width = 1
        
        # Highlight new edges
        for edge_key in all_highlighted_edges:
            if edge_key in edge_map and edge_map[edge_key] < len(figure.data):
                trace_idx = edge_map[edge_key]
                
                if edge_key in incoming_edges:
                    figure.data[trace_idx].line.color = incoming_edge_color
                    figure.data[trace_idx].line.width = 2  # Thinner edges
                elif edge_key in outgoing_edges:
                    figure.data[trace_idx].line.color = outgoing_edge_color  
                    figure.data[trace_idx].line.width = 2  # Thinner edges
    
    def _patch_all_edges_to_default(self, patch, graph_data, default_edge_color):
        """Reset all edges to default color efficiently."""
        edge_map = self._create_edge_mapping(graph_data)
        
        # Reset all existing edges to default
        for trace_idx in edge_map.values():
            patch['data'][trace_idx]['line']['color'] = default_edge_color
            patch['data'][trace_idx]['line']['width'] = 1
    
    def _reset_specific_edges(self, patch, edges_to_reset, default_edge_color):
        """Reset only specific edges to default color."""
        edge_map = self._create_edge_mapping(self.get_graph_data())
        
        for edge_key in edges_to_reset:
            if edge_key in edge_map:
                trace_idx = edge_map[edge_key]
                patch['data'][trace_idx]['line']['color'] = default_edge_color
                patch['data'][trace_idx]['line']['width'] = 1
    
    def _patch_edge_colors(self, patch, incoming_edges, outgoing_edges, graph_data):
        """Efficiently patch edge colors without O(N²) operations."""
        # ENHANCED EDGE COLORS - Much thicker and more visible
        incoming_edge_color = 'rgba(34, 197, 94, 0.9)'    # Bright green, more opaque
        outgoing_edge_color = 'rgba(59, 130, 246, 0.9)'   # Bright blue, more opaque
        
        # Create a mapping of (i,j) -> edge_trace_index for efficient lookup
        edge_map = self._create_edge_mapping(graph_data)
        
        # Update only the edges we care about (much faster)
        all_highlighted_edges = incoming_edges | outgoing_edges
        
        for edge_key in all_highlighted_edges:
            if edge_key in edge_map:
                trace_idx = edge_map[edge_key]
                
                if edge_key in incoming_edges:
                    color = incoming_edge_color
                    width = 3  # Thinner edges for cleaner look
                elif edge_key in outgoing_edges:
                    color = outgoing_edge_color
                    width = 3  # Thinner edges for cleaner look
                else:
                    color = 'rgba(0, 0, 0, 0.05)'  # Original default
                    width = 1
                
                # Patch the specific edge trace
                patch['data'][trace_idx]['line']['color'] = color
                patch['data'][trace_idx]['line']['width'] = width
    
    def _create_edge_mapping(self, graph_data):
        """Create mapping from (i,j) to edge trace index - computed once and cached."""
        if not hasattr(self, '_edge_mapping'):
            self._edge_mapping = {}
            trace_idx = 1  # Start after node trace
            
            # This follows the same logic as the edge creation
            for i in range(len(graph_data.nodes)):
                for j in range(i + 1, len(graph_data.nodes)):
                    # Only include edges that actually exist in the graph
                    if (i < len(graph_data.adjacency_matrix) and 
                        j < len(graph_data.adjacency_matrix[0]) and
                        graph_data.adjacency_matrix[i, j] != 0):
                        
                        # Map both directions to the same trace
                        self._edge_mapping[(i, j)] = trace_idx
                        self._edge_mapping[(j, i)] = trace_idx
                        trace_idx += 1
            
            # Add edges to logit node (last node is logit node)
            if len(graph_data.nodes) > 0:
                logit_idx = len(graph_data.nodes) - 1
                logit_col_idx = graph_data.adjacency_matrix.shape[1] - 1
                
                for i in range(len(graph_data.nodes) - 1):  # Exclude logit node itself
                    node = graph_data.nodes[i]
                    if (node.node_id_original < len(graph_data.adjacency_matrix) and
                        graph_data.adjacency_matrix[node.node_id_original, logit_col_idx] != 0):
                        
                        # Map edge to logit
                        self._edge_mapping[(i, logit_idx)] = trace_idx
                        self._edge_mapping[(i, "logit")] = trace_idx  # Special mapping
                        trace_idx += 1
        
        return self._edge_mapping
    
    def _create_fast_highlight_figure(self, selected_feature_id: int, graph_data):
        """Fast highlighting by updating colors without deep copy."""
        import plotly.graph_objects as go
        
        # PERFORMANCE: Use shallow operations instead of expensive deep copy
        figure = go.Figure(self._static_figure)
        
        # Get connected nodes and edges efficiently
        highlighted_nodes = self._get_highlighted_nodes(graph_data, selected_feature_id)
        incoming_edges, outgoing_edges = self._get_highlighted_edges(graph_data, selected_feature_id)
        
        # Update node colors (first trace is always nodes)
        if figure.data and len(figure.data) > 0:
            node_trace = figure.data[0]
            
            # Create new color array efficiently
            colors = ['#f8f9fa'] * len(graph_data.nodes)  # Light gray fill
            
            # Set highlighted colors
            colors[selected_feature_id] = '#EF4444'  # Red for selected
            for node_id in highlighted_nodes:
                if node_id != selected_feature_id and node_id < len(colors):
                    colors[node_id] = '#F59E0B'  # Orange for connected
            
            # Update efficiently
            node_trace.marker.color = colors
            
            # Update edge colors efficiently (traces after node trace)
            self._update_edge_colors(figure, incoming_edges, outgoing_edges)
        
        return figure
    
    def _get_highlighted_edges(self, graph_data, selected_feature_id):
        """Get edges connected to selected feature (ULTRA-FAST using edge cache)."""
        if self.edge_cache.is_cached():
            # FAST PATH: Use precomputed edge cache (O(1) lookup)
            return self.edge_cache.get_highlighted_edge_sets(selected_feature_id)
        else:
            # FALLBACK: Use slow matrix traversal (should rarely happen)
            print("⚠️  Edge cache not available, using slow matrix traversal")
            return self._get_highlighted_edges_slow(graph_data, selected_feature_id)
    
    def _get_highlighted_edges_slow(self, graph_data, selected_feature_id):
        """Fallback slow method for getting highlighted edges."""
        incoming_edges = set()
        outgoing_edges = set()
        
        if selected_feature_id < len(graph_data.adjacency_matrix):
            adj_matrix = graph_data.adjacency_matrix
            
            # Check only selected node's row and column (O(N) instead of O(N²))
            # Use adj_matrix.shape[1] to include the final logit column
            for i in range(adj_matrix.shape[1]):  # Include final logit column
                # Outgoing edges from selected node
                if i < adj_matrix.shape[1] and adj_matrix[selected_feature_id, i] != 0:
                    outgoing_edges.add((selected_feature_id, i))
            
            # Incoming edges to selected node (only from other features, not from logit)
            for i in range(adj_matrix.shape[0]):  # Only check feature rows
                if adj_matrix[i, selected_feature_id] != 0:
                    incoming_edges.add((i, selected_feature_id))
        
        return incoming_edges, outgoing_edges
    
    def _update_edge_colors(self, figure, incoming_edges, outgoing_edges):
        """Update edge colors efficiently without recreation."""
        # SIMPLIFIED: For maximum speed, skip edge highlighting for now
        # Edge highlighting would require mapping each trace to its (i,j) pair
        # which is complex without recreating the edge logic
        
        # TODO: Implement efficient edge highlighting by:
        # 1. Storing edge metadata during static figure creation
        # 2. Using that metadata to quickly find and update relevant edge traces
        
        # For now, users get fast node highlighting, edges stay default
        pass
    
    def _get_highlighted_nodes(self, graph_data, selected_feature_id):
        """Get nodes connected to selected feature (fast implementation)."""
        highlighted_nodes = set()
        highlighted_nodes.add(selected_feature_id)
        
        if selected_feature_id < len(graph_data.adjacency_matrix):
            # Only check connections for selected node, not all pairs
            adj_row = graph_data.adjacency_matrix[selected_feature_id]
            adj_col = graph_data.adjacency_matrix[:, selected_feature_id]
            
            # Outgoing edges
            for i, weight in enumerate(adj_row):
                if weight != 0 and i < len(graph_data.nodes):
                    highlighted_nodes.add(i)
            
            # Incoming edges  
            for i, weight in enumerate(adj_col):
                if weight != 0 and i < len(graph_data.nodes):
                    highlighted_nodes.add(i)
        
        return highlighted_nodes
    
    def create_graph_container(self) -> html.Div:
        """Create the graph container component that fits properly."""
        # Create minimal initial figure without precomputation
        graph_data = self.get_graph_data()
        initial_figure = self.graph_renderer.create_figure(
            graph_data, 
            selected_feature_id=None, 
            nodes_with_descriptions=set(),
            node_to_cluster={},
            cluster_highlighted_nodes=set(),
            edge_cache=self.edge_cache
        )
        
        return html.Div([
            # Control buttons (top-left overlay)
            html.Div([
                html.Button(
                    "Add Text",
                    id="annotation-mode-btn",
                    style={
                        'backgroundColor': '#fefce8',
                        'color': '#ca8a04',
                        'border': '1px solid #fed7aa',
                        'padding': '6px 14px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '11px',
                        'fontWeight': '600',
                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                        'minWidth': '90px',
                        'textAlign': 'center',
                        'marginRight': '8px'
                    }
                ),
                html.Button(
                    "Save State",
                    id="save-state-btn",
                    style={
                        'backgroundColor': '#dcfce7',
                        'color': '#16a34a',
                        'border': '1px solid #bbf7d0',
                        'padding': '6px 14px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '11px',
                        'fontWeight': '600',
                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                        'minWidth': '90px',
                        'textAlign': 'center',
                        'marginRight': '8px'
                    }
                ),
                html.Button(
                    "Load State",
                    id="load-state-btn",
                    style={
                        'backgroundColor': '#dbeafe',
                        'color': '#2563eb',
                        'border': '1px solid #93c5fd',
                        'padding': '6px 14px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontSize': '11px',
                        'fontWeight': '600',
                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                        'minWidth': '90px',
                        'textAlign': 'center'
                    }
                ),
                # Hidden file input for loading
                dcc.Upload(
                    id='load-state-upload',
                    children=html.Div(),
                    style={
                        'position': 'absolute',
                        'top': '0',
                        'left': '0',
                        'width': '1px',
                        'height': '1px',
                        'opacity': '0',
                        'pointerEvents': 'none'
                    },
                    accept='.json',
                    multiple=False
                )
            ], style={
                'position': 'absolute',
                'top': '12px',
                'left': '12px',
                'zIndex': '1000',
                'display': 'flex',
                'alignItems': 'center'
            }),
            
            # Mode indicator (top-center overlay)
            html.Div(
                id="annotation-mode-indicator",
                style={
                    'position': 'absolute',
                    'top': '12px',
                    'left': '50%',
                    'transform': 'translateX(-50%)',
                    'padding': '6px 12px',
                    'fontSize': '10px',
                    'backgroundColor': '#dbeafe',
                    'color': '#1e40af',
                    'borderRadius': '4px',
                    'zIndex': '1000',
                    'display': 'none'
                }
            ),
            
            # Main graph
            dcc.Graph(
                id='active-feature-graph',
                figure=initial_figure,  # Use the pre-computed initial figure
                style={
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': self.config.graph.background_color
                },
                config={
                    'scrollZoom': False,
                    'displayModeBar': False,
                    'responsive': True
                }
            ),
            
            # Text annotations overlay container
            html.Div(
                id='text-annotations-container',
                children=[],
                style={
                    'position': 'absolute',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'pointerEvents': 'none',
                    'zIndex': '999'
                }
            ),
            
            # Hidden trigger for annotation position updates
            html.Div(
                id='annotation-position-trigger',
                children=[],
                style={'display': 'none'}
            )
        ], style={
            'width': '100%',
            'height': '100%',
            'backgroundColor': self.config.graph.background_color,
            'fontFamily': 'Inter, Arial, sans-serif',
            'overflow': 'hidden',
            'position': 'relative'
        })
    
    def _create_edges(self, base_data: GraphData, nodes) -> list:
        """Create edge list from adjacency matrix including feature-to-logit edges."""
        edges = []

        # 1. Feature-to-feature edges
        for i, node_i in enumerate(nodes[:-1]):  # Exclude logit node
            for j, node_j in enumerate(nodes[:-1]):  # Exclude logit node
                if i != j and (i < len(base_data.adjacency_matrix) and
                             j < len(base_data.adjacency_matrix[0]) - 1):  # Exclude logit column
                    weight = base_data.adjacency_matrix[
                        node_i.node_id_original, node_j.node_id_original
                    ]
                    if weight != 0.0:
                        edges.append(FeatureEdge(
                            from_node=i,
                            to_node=j,
                            weight=weight
                        ))

        # 2. Feature-to-logit edges (from feature nodes to the last logit node)
        if len(nodes) > 0:
            logit_node_idx = len(nodes) - 1
            logit_col_idx = base_data.adjacency_matrix.shape[1] - 1

            for i, node_i in enumerate(nodes[:-1]):  # Exclude logit node itself
                if (i < len(base_data.adjacency_matrix) and
                    base_data.adjacency_matrix[node_i.node_id_original, logit_col_idx] != 0):

                    weight = base_data.adjacency_matrix[node_i.node_id_original, logit_col_idx]
                    edges.append(FeatureEdge(
                        from_node=i,
                        to_node=logit_node_idx,
                        weight=weight
                    ))

        return edges
    
    def get_feature_by_id(self, feature_id: int):
        """Get feature node by ID."""
        graph_data = self.get_graph_data()
        if 0 <= feature_id < len(graph_data.nodes):
            return graph_data.nodes[feature_id]
        return None
    
    def clear_cache(self):
        """Clear cached graph data."""
        self._graph_data = None
        self._static_figure = None
        self._last_highlighted = None
        self._original_colors = None
        # Clear pre-computed figures
        self._precomputed_figures.clear()
        self._precomputation_complete = False
        self._precomputation_progress = 0
        # Clear edge cache
        self.edge_cache.clear_cache()
        if hasattr(self, '_edge_mapping'):
            delattr(self, '_edge_mapping')
        self.data_loader.clear_cache()
