from typing import List

from ...data.models import GraphData, FeatureNode
from ...data.loaders import DataLoader
from ...config.settings import GraphConfig

class GraphLayoutCalculator:
    """Calculates optimal graph layout based on content and screen size."""
    
    def __init__(self, config: GraphConfig, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        # Default screen size - updated for 80% width
        self.screen_width = 1280
        self.screen_height = 900
    
    def set_screen_dimensions(self, width: int, height: int):
        """Set screen dimensions for layout calculations."""
        self.screen_width = width
        self.screen_height = height
    
    def calculate_node_positions(self, graph_data: GraphData) -> List[FeatureNode]:
        """Calculate responsive positions for all nodes with proper alignment."""
        
        # Calculate flexible column widths and layout that matches renderer
        layout_config = self._calculate_flexible_layout(graph_data)
        
        # Group features by layer and position
        layer_position_features = {}
        for i, (ctx_pos, layer, feat_idx) in enumerate(graph_data.feature_indices):
            key = (int(layer), int(ctx_pos))
            if key not in layer_position_features:
                layer_position_features[key] = []
            layer_position_features[key].append((i, feat_idx))

        # Feature grouping and layout config calculated
        
        # Create nodes with proper positioning that aligns with grid
        nodes = []
        for layer in range(graph_data.n_layers):
            # Calculate y position to align with grid
            y = layout_config['start_y'] + (layer + 0.5) * layout_config['layer_height']
            
            for pos in range(graph_data.prompt_length):
                features = layer_position_features.get((layer, pos), [])
                if not features:
                    continue
                
                # Get the center position for this token
                center_x = layout_config['token_x_positions'][pos]
                
                # Calculate x positions for features in this cell using optimal spacing
                x_positions = self._calculate_feature_positions_in_cell_uniform(
                    center_x, len(features), layout_config
                )
                
                for i, (node_id, feature_idx) in enumerate(features):
                    feature_config = self.data_loader.load_feature_dict(layer, feature_idx)
                    feature_desc = (feature_config.get("description", f"Feature {feature_idx}")
                                  if feature_config else f"Feature {feature_idx}")

                    # Get frequency for this feature
                    feature_frequency = None
                    if graph_data.feature_frequencies:
                        feature_key = (pos, layer, feature_idx)
                        feature_frequency = graph_data.feature_frequencies.get(feature_key)

                    node = FeatureNode(
                        id=node_id,
                        x=x_positions[i],
                        y=y,
                        layer=layer,
                        pos=pos,
                        feature_idx=feature_idx,
                        token=graph_data.input_tokens[pos] if pos < len(graph_data.input_tokens) else f"Token{pos}",
                        description=feature_desc,
                        config=feature_config,
                        node_id_original=node_id,
                        frequency=feature_frequency
                    )
                    nodes.append(node)
                    # Node positioned
        
        # Add embedding token nodes (small squares above regular tokens)
        if graph_data.embedding_adjacency is not None:
            embedding_nodes = self._create_embedding_nodes(
                len(nodes), graph_data, layout_config
            )
            nodes.extend(embedding_nodes)

        # Add logit node
        logit_node = self._create_logit_node(
            len(nodes),
            layout_config['token_x_positions'][-1],
            layout_config['start_y'] + graph_data.n_layers * layout_config['layer_height']
        )
        nodes.append(logit_node)

        # Update graph_data with calculated positions
        graph_data.token_x_positions = layout_config['token_x_positions']

        return nodes
    
    def _calculate_flexible_layout(self, graph_data: GraphData) -> dict:
        """Calculate layout with spacing based on densest layer to prevent overflow."""
        
        node_diameter = 12
        
        # Use same screen calculations
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
        
        # Max nodes in any cell: {max_nodes_in_any_cell}
        
        # STEP 2: Calculate node spacing that works for the densest cell
        # Assume minimum column width and calculate spacing
        min_column_width = 80  # Minimum reasonable column width
        padding = 20  # Total padding per column
        available_space_per_column = min_column_width - padding
        
        if max_nodes_in_any_cell <= 1:
            node_spacing = 20  # Default spacing when not constrained
        else:
            # Calculate spacing: (available_space - node_diameter) / (n_nodes - 1)
            node_spacing = (available_space_per_column - node_diameter) / (max_nodes_in_any_cell - 1)
            node_spacing = max(16, min(20, node_spacing))  # Clamp between 16-20px
        
        # Calculated node spacing: {node_spacing}
        
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
        
        # Final layout calculated
        
        return {
            'column_widths': column_widths,
            'layer_height': layer_height,
            'content_width': content_width,
            'content_height': content_height,
            'start_x': start_x,
            'start_y': start_y,
            'token_x_positions': token_x_positions,
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
            print(f"Column {pos}: max {max_nodes_in_column} nodes in any single layer")
        
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
        
        print(f"Maximum nodes in any single cell: {max_nodes}")
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
    
    def _calculate_optimal_spacing(self, column_width: float, num_nodes: int) -> float:
        """Calculate optimal spacing for nodes within a specific column width."""
        if num_nodes <= 1:
            return 0
        
        node_diameter = 12
        padding = 20  # Total padding for the column
        available_width = column_width - padding
        
        # Calculate spacing: (available_width - node_diameter) / (num_nodes - 1)
        if num_nodes > 1:
            spacing = (available_width - node_diameter) / (num_nodes - 1)
            return max(16, spacing)  # Minimum 16px spacing for visual appeal
        return 20  # Default spacing
    
    def _calculate_feature_positions_in_cell_uniform(self, center_x: float, 
                                                  num_features: int, 
                                                  layout_config: dict) -> List[float]:
        """Calculate x positions with optimal spacing centered in the cell."""
        if num_features == 1:
            return [center_x]
        
        # Use the optimal node spacing from layout config
        node_spacing = layout_config['node_spacing']
        
        # Calculate positions centered around center_x
        total_span = (num_features - 1) * node_spacing
        start_x = center_x - total_span / 2
        
        positions = []
        for i in range(num_features):
            x_pos = start_x + i * node_spacing
            positions.append(x_pos)
        
        return positions
    
    def _calculate_feature_positions_in_cell(self, center_x: float, 
                                           num_features: int, 
                                           column_width: float) -> List[float]:
        """Legacy method - kept for compatibility."""
        # Use default spacing config
        default_config = {'node_spacing': 20, 'node_diameter': 12}
        return self._calculate_feature_positions_in_cell_uniform(center_x, num_features, default_config)
    
    def _create_embedding_nodes(self, starting_node_id: int, graph_data: GraphData, layout_config: dict) -> List[FeatureNode]:
        """Create embedding token nodes as small squares above regular tokens."""
        embedding_nodes = []

        # Position embedding nodes at the same level as the top of the graph
        embedding_y = layout_config['start_y']  # Same level as the top of the graph

        for pos in range(1, graph_data.prompt_length):  # Skip token 0
            # Get the center position for this token column
            center_x = layout_config['token_x_positions'][pos]

            # Create embedding node for this token position
            embedding_node = FeatureNode(
                id=starting_node_id + pos - 1,  # Adjust ID since we skip token 0
                x=center_x,
                y=embedding_y,
                layer="embedding",
                pos=pos,
                feature_idx="embedding",
                token=graph_data.input_tokens[pos] if pos < len(graph_data.input_tokens) else f"Token{pos}",
                description=f"Embedding for token '{graph_data.input_tokens[pos] if pos < len(graph_data.input_tokens) else f'Token{pos}'}'",
                config=None,
                node_id_original=starting_node_id + pos - 1,  # Adjust ID since we skip token 0
                frequency=None
            )
            embedding_nodes.append(embedding_node)

        return embedding_nodes

    def _create_logit_node(self, node_id: int, x: float, y: float) -> FeatureNode:
        """Create the logit output node."""
        return FeatureNode(
            id=node_id,
            x=x,
            y=y,
            layer="logit",
            pos="logit",
            feature_idx="logit",
            token="logit",
            description="Final logit output",
            config=None,
            node_id_original=node_id,
            frequency=None  # Logit nodes don't have frequencies
        )
