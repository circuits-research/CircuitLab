import plotly.graph_objects as go
from typing import List, Optional, Set

from ...data.models import GraphData
from ...config.settings import GraphConfig
from .nodes import NodeRenderer
from .edges import EdgeRenderer

class GraphRenderer:
    """Main graph renderer that combines all components."""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.node_renderer = NodeRenderer(config)
        self.edge_renderer = EdgeRenderer(config)
        # Screen dimensions - will be set based on viewport (80% of screen now)
        self.screen_width = 1280
        self.screen_height = 900
        
        # Get background color with fallback
        self.background_color = getattr(config, 'background_color', '#fcf8f3')
    
    def set_screen_dimensions(self, width: int, height: int):
        """Set screen dimensions for responsive calculations."""
        self.screen_width = width
        self.screen_height = height
    
    def create_figure(self, graph_data: GraphData, 
                     selected_feature_id: Optional[int] = None,
                     nodes_with_descriptions: Set[int] = None,
                     node_to_cluster: dict = None,
                     cluster_highlighted_nodes: Set[int] = None,
                     edge_cache = None) -> go.Figure:
        """Create the complete graph figure with cluster coloring."""
        
        nodes_with_descriptions = nodes_with_descriptions or set()
        cluster_highlighted_nodes = cluster_highlighted_nodes or set()
        
        # Create figure with cluster highlighting
        
        # Creating graph figure
        
        # Get highlighted nodes (connected to selected feature)
        highlighted_nodes = set()
        if selected_feature_id is not None:
            highlighted_nodes = self._get_highlighted_nodes(graph_data, selected_feature_id)
        
        # Get intersection nodes to color red
        intersection_nodes = set()
        if graph_data.feature_list_intersection is not None:
            # feature_list_intersection contains tuples of (pos, layer, idx)
            # Need to find corresponding node indices in graph_data.nodes
            for pos, layer, idx in graph_data.feature_list_intersection:
                for node_i, node in enumerate(graph_data.nodes):
                    # Skip logit and embedding nodes for intersection feature matching
                    if node.pos == "logit" or node.layer == "logit" or node.layer == "embedding":
                        continue
                    if (int(node.pos) == int(pos) and
                        int(node.layer) == int(layer) and
                        int(node.feature_idx) == int(idx)):
                        intersection_nodes.add(node_i)
                        break
        
        # Create node traces with cluster coloring (returns list of traces)
        node_traces = self.node_renderer.create_node_trace(
            graph_data.nodes,
            selected_feature_id,
            highlighted_nodes,
            nodes_with_descriptions,
            node_to_cluster,
            cluster_highlighted_nodes,
            intersection_nodes
        )
        
        # Create edge traces with proper parameters and null check
        edge_traces = []
        if graph_data.adjacency_matrix is not None:
            edge_traces = self.edge_renderer.create_edge_traces(
                graph_data.nodes,
                graph_data.adjacency_matrix,
                selected_feature_id,
                edge_cache
            )

        # Create embedding edge traces if embedding adjacency exists (disabled by default)
        embedding_edge_traces = []
        if self.config.show_embedding_edges and graph_data.embedding_adjacency is not None:
            embedding_edge_traces = self._create_embedding_edge_traces(
                graph_data.nodes, graph_data.embedding_adjacency,
                selected_feature_id, highlighted_nodes
            )
        
        # Create layout config for all components
        layout_config = self._calculate_responsive_layout(graph_data)
        
        # Create all layout elements
        grid_traces = self._create_flexible_grid_traces(graph_data, layout_config)
        token_labels = self._create_token_labels(graph_data, layout_config)
        description_traces = self._create_multiple_descriptions(graph_data, layout_config, nodes_with_descriptions)
        logit_traces = self._create_logit_elements(graph_data, layout_config)
        
        # Combine all traces - nodes LAST so they render on top of edges
        data = edge_traces + embedding_edge_traces + grid_traces + node_traces + token_labels + description_traces + logit_traces
        
        # Create layout
        layout = self._create_layout(graph_data, layout_config)
        
        return go.Figure(data=data, layout=layout)
    
    def _calculate_responsive_layout(self, graph_data: GraphData) -> dict:
        """Calculate layout with spacing based on densest layer to prevent overflow."""
        
        node_diameter = 12
        
        # Use screen space efficiently
        usable_width = self.screen_width * 0.90
        usable_height = self.screen_height * 0.85
        margin_x = 40
        margin_y = 50
        available_width = usable_width - (2 * margin_x)
        available_height = usable_height - (2 * margin_y)
        
        # STEP 1: Find the maximum nodes in any single cell (layer, position)
        max_nodes_in_any_cell = 0
        for layer in range(graph_data.n_layers):
            for pos in range(graph_data.prompt_length):
                nodes_in_cell = 0
                for ctx_pos, node_layer, feat_idx in graph_data.feature_indices:
                    if int(node_layer) == layer and int(ctx_pos) == pos:
                        nodes_in_cell += 1
                max_nodes_in_any_cell = max(max_nodes_in_any_cell, nodes_in_cell)
        
        # Max nodes in any cell calculated
        
        # STEP 2: Calculate node spacing that works for the densest cell
        min_column_width = 80  # Minimum reasonable column width
        padding = 20  # Total padding per column
        available_space_per_column = min_column_width - padding
        
        if max_nodes_in_any_cell <= 1:
            node_spacing = 20  # Default spacing when not constrained
        else:
            # Calculate spacing: (available_space - node_diameter) / (n_nodes - 1)
            node_spacing = (available_space_per_column - node_diameter) / (max_nodes_in_any_cell - 1)
            node_spacing = max(16, min(20, node_spacing))  # Clamp between 16-20px
        
        # Node spacing calculated
        
        # STEP 3: Calculate column widths based on this spacing
        column_widths = []
        for pos in range(graph_data.prompt_length):
            max_nodes_in_column = self._calculate_max_nodes_in_column_single(graph_data, pos)
            
            if max_nodes_in_column == 0:
                column_width = 40  # Reduced width for empty columns
            elif max_nodes_in_column == 1:
                column_width = node_diameter + padding
            else:
                # Use the calculated spacing
                nodes_width = (max_nodes_in_column - 1) * node_spacing + node_diameter
                column_width = nodes_width + padding
            
            # Allow smaller width for empty columns, minimum 60 for columns with nodes
            min_width = 40 if max_nodes_in_column == 0 else 60
            column_widths.append(max(min_width, column_width))
        
        # Scale down if total width exceeds available space
        total_width = sum(column_widths)
        if total_width > available_width:
            scale_factor = available_width / total_width
            column_widths = [w * scale_factor for w in column_widths]
            # Also scale down node spacing proportionally
            node_spacing = node_spacing * scale_factor
        
        # Calculate layout dimensions
        content_width = sum(column_widths)
        layer_height = max(50, available_height / graph_data.n_layers)
        content_height = layer_height * graph_data.n_layers
        start_x = -content_width / 2
        start_y = -content_height / 2
        
        # Calculate token positions (centered in each column)
        token_x_positions = []
        current_x = start_x
        for i in range(graph_data.prompt_length):
            token_x = current_x + column_widths[i] / 2
            token_x_positions.append(token_x)
            current_x += column_widths[i]
        
        # Calculate nodes per column for compatibility
        nodes_per_column = self._calculate_nodes_per_column(graph_data)
        
        # Layout calculations completed
        
        return {
            'column_widths': column_widths,
            'layer_height': layer_height,
            'content_width': content_width,
            'content_height': content_height,
            'start_x': start_x,
            'start_y': start_y,
            'token_x_positions': token_x_positions,
            'margin_x': margin_x,
            'margin_y': margin_y,
            'nodes_per_column': nodes_per_column,
            'node_spacing': node_spacing,
            'node_diameter': node_diameter
        }
    
    def _calculate_nodes_per_column(self, graph_data: GraphData) -> List[int]:
        """Calculate the MAXIMUM number of nodes in any single layer within each column."""
        # Initialize with zeros
        max_nodes_per_column = [0] * graph_data.prompt_length
        
        # For each column (token position)
        for pos in range(graph_data.prompt_length):
            max_nodes_in_column = 0
            
            # For each layer, count nodes at this position
            for layer in range(graph_data.n_layers):
                nodes_in_this_layer = 0
                for ctx_pos, node_layer, feat_idx in graph_data.feature_indices:
                    if int(node_layer) == layer and int(ctx_pos) == pos:
                        nodes_in_this_layer += 1
                
                # Track the maximum nodes in any single layer for this column
                max_nodes_in_column = max(max_nodes_in_column, nodes_in_this_layer)
            
            max_nodes_per_column[pos] = max_nodes_in_column
            # Column analysis completed
        
        return max_nodes_per_column
    
    def _calculate_max_nodes_in_any_cell(self, graph_data: GraphData) -> int:
        """Find the maximum number of nodes in any single cell (layer, position) in the entire graph."""
        max_nodes = 0
        
        for layer in range(graph_data.n_layers):
            for pos in range(graph_data.prompt_length):
                nodes_in_cell = 0
                for ctx_pos, node_layer, feat_idx in graph_data.feature_indices:
                    if int(node_layer) == layer and int(ctx_pos) == pos:
                        nodes_in_cell += 1
                max_nodes = max(max_nodes, nodes_in_cell)
        
        # Maximum nodes calculated
        return max_nodes
    
    def _calculate_max_nodes_in_column_single(self, graph_data: GraphData, pos: int) -> int:
        """Calculate the maximum nodes in any single layer for a specific column."""
        max_nodes = 0
        
        for layer in range(graph_data.n_layers):
            nodes_in_this_layer = 0
            for ctx_pos, node_layer, feat_idx in graph_data.feature_indices:
                if int(node_layer) == layer and int(ctx_pos) == pos:
                    nodes_in_this_layer += 1
            max_nodes = max(max_nodes, nodes_in_this_layer)
        
        return max_nodes
    
    def _calculate_optimal_node_spacing(self) -> int:
        """Calculate optimal spacing between nodes for visual appeal."""
        return 20  # 20px center-to-center spacing looks good visually
    
    def _create_flexible_grid_traces(self, graph_data: GraphData, layout_config: dict) -> List[go.Scatter]:
        """Create grid lines with flexible column widths."""
        grid_traces = []
        
        # Vertical lines between token columns (flexible widths)
        current_x = layout_config['start_x']
        for pos in range(graph_data.prompt_length + 1):
            grid_traces.append(go.Scatter(
                x=[current_x, current_x],
                y=[layout_config['start_y'], layout_config['start_y'] + layout_config['content_height']],
                mode='lines',
                line=dict(width=0.5, color='rgba(148, 163, 184, 0.3)'),
                hoverinfo='none',
                showlegend=False
            ))
            if pos < graph_data.prompt_length:
                current_x += layout_config['column_widths'][pos]
        
        # Horizontal lines between layers
        for layer in range(graph_data.n_layers + 1):
            y_pos = layout_config['start_y'] + layer * layout_config['layer_height']
            grid_traces.append(go.Scatter(
                x=[layout_config['start_x'], layout_config['start_x'] + layout_config['content_width']],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(width=0.5, color='rgba(148, 163, 184, 0.2)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Layer labels
        for layer in range(graph_data.n_layers):
            y_pos = layout_config['start_y'] + (layer + 0.5) * layout_config['layer_height']
            x_pos = layout_config['start_x'] - 20
            
            grid_traces.append(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='text',
                text=[f"<b>L{layer}<b>"],
                textfont=dict(size=12, color='black', family="Inter, Arial"),
                textposition="middle center",
                showlegend=False,
                hoverinfo='skip'
            ))
        
        return grid_traces
    
    def _create_token_labels(self, graph_data: GraphData, layout_config: dict) -> List[go.Scatter]:
        """Create token labels centered in each column with vertical orientation."""
        # Store annotations in the layout instead of creating scatter traces
        annotations = []
        
        for i in range(graph_data.prompt_length):
            if 0 < i < len(graph_data.input_tokens):
                token = graph_data.input_tokens[i]
                # Use the centered position for this column
                x_pos = layout_config['token_x_positions'][i]
                y_pos = layout_config['start_y'] - 25
                
                # Scale font size based on available space (smaller font)
                column_width = layout_config['column_widths'][i]
                font_size = 15
                
                # Create annotation with text rotation
                annotation = dict(
                    x=x_pos,
                    y=y_pos,
                    text=f"<b>{token}</b>",
                    showarrow=False,
                    font=dict(size=font_size, color='black', family="Inter, Arial"),
                    textangle=-30,  # Rotate text 75 degrees for more vertical orientation
                    xanchor='center',
                    yanchor='middle'
                )
                annotations.append(annotation)
        
        # Store annotations for later use in layout
        layout_config['token_annotations'] = annotations
        
        return []  # Return empty list since we're using annotations instead
    
    def _create_multiple_descriptions(self, graph_data: GraphData, layout_config: dict, 
                                    nodes_with_descriptions: Set[int]) -> List[go.Scatter]:
        """Create feature descriptions for all clicked nodes."""
        description_traces = []
        
        for node_id in nodes_with_descriptions:
            if node_id < len(graph_data.nodes):
                node = graph_data.nodes[node_id]
                
                # Only show description if node has one
                if hasattr(node, 'description') and node.description:
                    # Truncate long descriptions
                    desc = node.description
                    if len(desc) > 45:
                        desc = desc[:42] + "..."
                    
                    # Position description below the node
                    desc_y = node.y - 20
                    
                    description_traces.append(go.Scatter(
                        x=[node.x],
                        y=[desc_y],
                        mode='text',
                        text=[f"<b>{desc}</b>"],
                        textfont=dict(size=10, color='#1f2937', family="Inter, Arial"),
                        textposition="middle center",
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        return description_traces
    
    def _create_logit_elements(self, graph_data: GraphData, layout_config: dict) -> List[go.Scatter]:
        """Create responsive arrow and text for logit."""
        if not graph_data.nodes:
            return []
        
        logit_node = graph_data.nodes[-1]  # Last node should be logit
        traces = []
        
        # Main top logit token (large and green)
        text_x = logit_node.x
        main_trace = go.Scatter(
            x=[text_x],
            y=[logit_node.y + 20],
            mode='text',
            text=[f"<b>{graph_data.top_logit_token}</b>"],
            textfont=dict(size=16, color='green', family="Inter, Arial"),
            hoverinfo='none',
            showlegend=False
        )
        traces.append(main_trace)
        
        # Add top 5 logits if available
        if (hasattr(graph_data, 'top5_logit_tokens') and graph_data.top5_logit_tokens is not None and 
            hasattr(graph_data, 'top5_logit_probs') and graph_data.top5_logit_probs is not None):
            
            # Clean token display function
            def clean_token(token):
                # Remove weird symbols and clean up token display
                cleaned = token.strip()
                # Remove common weird symbols
                if cleaned.startswith('Ġ'):  # GPT-2 space marker
                    cleaned = cleaned[1:]
                if cleaned.startswith('▁'):  # SentencePiece space marker
                    cleaned = cleaned[1:]
                # Replace newlines and tabs with readable text
                cleaned = cleaned.replace('\n', '\\n').replace('\t', '\\t')
                # If token is just punctuation or weird chars, wrap in quotes for clarity
                if len(cleaned) == 1 and not cleaned.isalnum():
                    cleaned = f'"{cleaned}"'
                return cleaned if cleaned else '""'
            
            # Create main token (larger, prominent)
            main_token = clean_token(graph_data.top5_logit_tokens[0])
            main_prob = graph_data.top5_logit_probs[0]
            
            # Create text for main token (thick, black)
            main_text = f"<b style='color:#000000; font-size:16px; font-weight:700'>{main_token}</b>"
            main_prob_text = f"<span style='color:#000000; font-size:11px; font-weight:600'>{main_prob:.2%}</span>"
            
            # Create text for other tokens (thick, black)
            other_tokens = []
            for i in range(1, min(5, len(graph_data.top5_logit_tokens))):
                token = clean_token(graph_data.top5_logit_tokens[i])
                prob = graph_data.top5_logit_probs[i]
                other_tokens.append(f"<span style='font-weight:600'>{token} {prob:.1%}</span>")
            
            # Combine with clean spacing
            if other_tokens:
                other_text = f"<span style='color:#000000; font-size:10px; font-weight:600'>{' • '.join(other_tokens)}</span>"
                combined_text = f"{main_text} {main_prob_text}<br><span style='margin-top:2px'>{other_text}</span>"
            else:
                combined_text = f"{main_text} {main_prob_text}"
            
            # Replace the main trace with clean, centered display (moved down slightly)
            traces[0] = go.Scatter(
                x=[text_x],
                y=[logit_node.y + 19],  # Tiny adjustment up from +18
                mode='text',
                text=[combined_text],
                textfont=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
                hoverinfo='none',
                showlegend=False,
                textposition="middle center"
            )
        
        return traces
    
    def _create_layout(self, graph_data: GraphData, layout_config: dict) -> dict:
        """Create responsive layout that fills the screen properly."""
        
        # Calculate display ranges with padding
        padding = 40
        x_range = [
            layout_config['start_x'] - padding,
            layout_config['start_x'] + layout_config['content_width'] + padding
        ]
        y_range = [
            layout_config['start_y'] - padding,
            layout_config['start_y'] + layout_config['content_height'] + padding
        ]
        
        # Get token annotations if they exist
        annotations = layout_config.get('token_annotations', [])
        
        return go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=5, l=5, r=5, t=5),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=x_range,
                fixedrange=True
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=y_range,
                fixedrange=True
            ),
            width=self.screen_width,
            height=self.screen_height,
            plot_bgcolor=self.background_color,
            paper_bgcolor=self.background_color,
            clickmode='event+select',
            font=dict(family="Inter, Arial, sans-serif"),
            dragmode=False,
            autosize=False,
            annotations=annotations  # Add the rotated token annotations
        )
    
    def _get_highlighted_nodes(self, graph_data: GraphData, 
                              selected_feature_id: Optional[int]) -> set:
        """Get set of nodes that should be highlighted (only for double-clicked node)."""
        highlighted_nodes = set()
        
        if selected_feature_id is not None:
            # Getting highlighted nodes
            highlighted_nodes.add(selected_feature_id)
            
            # Add connected nodes based on adjacency matrix
            if selected_feature_id < len(graph_data.adjacency_matrix):
                # Nodes this feature affects (outgoing edges)
                outgoing = graph_data.adjacency_matrix[selected_feature_id] != 0
                for i, connected in enumerate(outgoing):
                    if connected and i < len(graph_data.nodes):
                        highlighted_nodes.add(i)

                # Nodes that affect this feature (incoming edges)
                incoming = graph_data.adjacency_matrix[:, selected_feature_id] != 0
                for i, connected in enumerate(incoming):
                    if connected and i < len(graph_data.nodes):
                        highlighted_nodes.add(i)

            # Also highlight embedding nodes that connect to this feature
            if graph_data.embedding_adjacency is not None and selected_feature_id < graph_data.embedding_adjacency.shape[1]:
                # Find embedding nodes that connect to the selected feature
                embedding_connections = graph_data.embedding_adjacency[:, selected_feature_id] > 0

                # Find the embedding nodes in the graph and highlight them
                for node_idx, node in enumerate(graph_data.nodes):
                    if node.layer == "embedding" and node.pos < len(embedding_connections):
                        if embedding_connections[node.pos]:
                            highlighted_nodes.add(node_idx)

            # Highlighted nodes calculated
        
        return highlighted_nodes

    def _create_embedding_edge_traces(self, nodes: List, embedding_adjacency,
                                     selected_feature_id: Optional[int] = None,
                                     highlighted_nodes: Set[int] = None) -> List[go.Scatter]:
        """Create edge traces from embedding tokens to feature nodes."""
        edge_traces = []

        # Find embedding nodes and feature nodes in order
        embedding_nodes = [node for node in nodes if node.layer == "embedding"]
        feature_nodes = [node for node in nodes if node.layer != "embedding" and node.layer != "logit"]

        if not embedding_nodes or not feature_nodes:
            return edge_traces

        # DEBUG: Show clear distinction between embedding tokens and feature nodes
        print("\n🔷 EMBEDDING EDGE DEBUG:")
        print(f"   📍 {len(embedding_nodes)} embedding tokens (squares above graph)")
        print(f"   📍 {len(feature_nodes)} feature nodes (circles in graph)")

        # DEBUG: Print exact coordinates of token 0 for comparison
        if embedding_nodes:
            token_0 = embedding_nodes[0]
            print(f"   🎯 TOKEN 0 COORDINATES: ({token_0.x:.1f}, {token_0.y:.1f}) - ID: {token_0.id}")
            print("   🎯 If you see an edge TO this coordinate, it's a mystery!")

        # Create separate traces using SAME colors as regular edge system
        normal_x_coords = []
        normal_y_coords = []
        incoming_x_coords = []  # For edges TO selected feature
        incoming_y_coords = []
        total_edges_created = 0
        highlighted_nodes = highlighted_nodes or set()

        for token_idx, embedding_node in enumerate(embedding_nodes):
            if token_idx < len(embedding_adjacency):
                embedding_row = embedding_adjacency[token_idx]
                token_edges = 0

                for feature_idx, edge_weight in enumerate(embedding_row):
                    if edge_weight > 0 and feature_idx < len(feature_nodes):  # Show if edge is active
                        feature_node = feature_nodes[feature_idx]
                        # STRICT CHECK: Never add edges from token 0
                        if token_idx == 0:
                            print(f"      🚨 BLOCKED edge from token 0 to feature {feature_idx} (weight: {edge_weight})")
                            continue

                        # Use EXACT same highlighting logic as regular edges
                        # Embedding edges are "incoming" to feature nodes when highlighted
                        is_incoming_to_selected = (feature_node.id == selected_feature_id)

                        if is_incoming_to_selected:
                            incoming_x_coords.extend([embedding_node.x, feature_node.x, None])
                            incoming_y_coords.extend([embedding_node.y, feature_node.y, None])
                        else:
                            normal_x_coords.extend([embedding_node.x, feature_node.x, None])
                            normal_y_coords.extend([embedding_node.y, feature_node.y, None])

                        token_edges += 1
                        total_edges_created += 1

                print(f"   🟩 EMBEDDING Token {token_idx} ('{embedding_node.token}'): {token_edges} edges to feature nodes")
                if token_idx == 0:
                    print(f"      ⚠️  Token 0 should have 0 edges - actual: {token_edges}")

        print(f"   📊 Total embedding edges created: {total_edges_created}")
        print("   🔗 Using SAME colors as regular edge system")

        # Create normal embedding edges trace (same color as regular normal edges)
        if normal_x_coords:
            edge_traces.append(go.Scatter(
                x=normal_x_coords,
                y=normal_y_coords,
                mode='lines',
                line=dict(width=1, color=self.config.normal_edge_color),  # SAME as regular edges
                hoverinfo='none',
                showlegend=False,
                name='Embedding Edges'
            ))

        # Create highlighted embedding edges trace (same color as regular incoming edges)
        if incoming_x_coords:
            edge_traces.append(go.Scatter(
                x=incoming_x_coords,
                y=incoming_y_coords,
                mode='lines',
                line=dict(width=3, color=self.config.incoming_edge_color),  # SAME as regular incoming edges
                hoverinfo='none',
                showlegend=False,
                name='Incoming Embedding Edges'
            ))

        return edge_traces
