from dash import Input, Output, State, no_update
from typing import Optional
import time

from ..visualization.components.graph_component import GraphComponent
from ..visualization.components.feature_display import FeatureDisplay

def register_callbacks(app, graph_component: GraphComponent):
    """Register all graph-related callbacks."""
    
    # PERFORMANCE: Remove problematic clientside callback, use optimized server-side
    # Clientside callback to detect viewport size
    app.clientside_callback(
        """
        function() {
            return {
                'width': window.innerWidth,
                'height': window.innerHeight
            };
        }
        """,
        Output('viewport-size-store', 'data'),
        Input('viewport-trigger', 'children')
    )
    
    # Add window resize listener
    app.clientside_callback(
        """
        function() {
            function updateViewport() {
                return {
                    'width': window.innerWidth,
                    'height': window.innerHeight
                };
            }
            
            // Update on load
            setTimeout(updateViewport, 100);
            
            // Update on resize
            window.addEventListener('resize', function() {
                setTimeout(updateViewport, 50);
            });
            
            return updateViewport();
        }
        """,
        Output('viewport-size-store', 'data', allow_duplicate=True),
        Input('viewport-trigger', 'id'),
        prevent_initial_call=True
    )
    
    # Server-side callback to update graph dimensions when viewport changes
    @app.callback(
        Output('active-feature-graph', 'figure', allow_duplicate=True),
        Input('viewport-size-store', 'data'),
        [State('selected-feature-store', 'data'),
         State('clicked-nodes-store', 'data'),
         State('clusters-store', 'data'),
         State('selected-cluster-store', 'data')],
        prevent_initial_call=True
    )
    def update_graph_dimensions(viewport_data, selected_feature, clicked_nodes_data, cluster_data, selected_clusters):
        """Update graph dimensions when viewport size changes."""
        if not viewport_data:
            return no_update
        
        # PERFORMANCE: Skip viewport updates if dimensions haven't changed significantly
        width = viewport_data.get('width', 1600)
        height = viewport_data.get('height', 1000)
        
        # Get current dimensions to check if update is needed
        current_width = getattr(graph_component, 'screen_width', 0)
        current_height = getattr(graph_component, 'screen_height', 0)
        
        # Only update if dimensions changed by more than 50px (debouncing)
        if (abs(width - current_width) < 50 and abs(height - current_height) < 50):
            return no_update
        
        # Update graph component dimensions
        graph_component.set_screen_dimensions(width, height)
        
        # Recreate graph with new dimensions
        nodes_with_descriptions = set()
        double_clicked_node = None
        cluster_highlighted_nodes = set()
        
        if clicked_nodes_data:
            nodes_with_descriptions = set(clicked_nodes_data.get('nodes_with_descriptions', []))
            double_clicked_node = clicked_nodes_data.get('double_clicked_node')
        
        # Process cluster highlighting
        if selected_clusters and cluster_data:
            if isinstance(selected_clusters, str):
                selected_clusters = [selected_clusters]
            elif not isinstance(selected_clusters, list):
                selected_clusters = []
                
            for cluster_id in selected_clusters:
                if cluster_id in cluster_data:
                    cluster_nodes = cluster_data[cluster_id].get('nodes', [])
                    for node_data in cluster_nodes:
                        node_index = node_data.get('node_index')
                        if node_index is not None:
                            cluster_highlighted_nodes.add(node_index)
        
        # Create updated figure
        figure = graph_component.create_graph_figure(
            selected_feature_id=double_clicked_node,
            nodes_with_descriptions=nodes_with_descriptions,
            cluster_data=cluster_data,
            cluster_highlighted_nodes=cluster_highlighted_nodes
        )
        
        return figure
    
    # SEPARATED: Handle ONLY node clicks on the main graph
    @app.callback(
        [Output('active-feature-graph', 'figure'),
         Output('activation-display', 'children'), 
         Output('selected-feature-store', 'data'),
         Output('clicked-nodes-store', 'data')],
        [Input('active-feature-graph', 'clickData')],  # ONLY node clicks
        [State('selected-feature-store', 'data'),
         State('clicked-nodes-store', 'data'),
         State('viewport-size-store', 'data')],
        prevent_initial_call=True
    )
    def handle_node_click_only(click_data, current_selection, clicked_nodes_data, viewport_data):
        """Handle ONLY individual node clicks on the main graph."""
        
        # NODE-ONLY: This callback only handles direct node clicks
        if not click_data:
            return no_update, no_update, no_update, no_update
        
        # Handle viewport dimensions (only if they changed significantly)
        if viewport_data:
            width = viewport_data.get('width', 1600)
            height = viewport_data.get('height', 1000)
            
            # Only update dimensions if they changed significantly
            current_width = getattr(graph_component, 'screen_width', 0)
            current_height = getattr(graph_component, 'screen_height', 0)
            
            if (abs(width * 0.68 - current_width) > 50 or abs(height - 96 - current_height) > 50):
                graph_component.set_screen_dimensions(width, height)
        
        # Initialize click data if None
        if clicked_nodes_data is None:
            clicked_nodes_data = {
                'nodes_with_descriptions': [],
                'double_clicked_node': None,
                'last_click_time': 0,
                'last_clicked_node': None
            }
        
        double_clicked_node = clicked_nodes_data.get('double_clicked_node')
        
        # Process node click
        if 'points' in click_data:
            point = click_data['points'][0]
            if 'pointIndex' in point:
                clicked_node_id = point['pointIndex']
                current_time = time.time()
                
                # Toggle node selection
                if double_clicked_node == clicked_node_id:
                    double_clicked_node = None  # Deselect
                else:
                    double_clicked_node = clicked_node_id  # Select
                
                # Update click data
                clicked_nodes_data.update({
                    'double_clicked_node': double_clicked_node,
                    'nodes_with_descriptions': [],
                    'last_click_time': current_time,
                    'last_clicked_node': clicked_node_id
                })
        
        # Create node highlighting (NO CLUSTERS - pure node highlighting)
        figure = graph_component.create_graph_figure(
            selected_feature_id=double_clicked_node,
            nodes_with_descriptions=set(),
            cluster_data=None,  # NO cluster data - pure node highlighting
            cluster_highlighted_nodes=set()
        )
        
        # Create feature display
        display_content = create_feature_display(graph_component, double_clicked_node)
        
        return figure, display_content, double_clicked_node, clicked_nodes_data

    # SEPARATED: Handle ONLY cluster highlighting changes
    @app.callback(
        Output('active-feature-graph', 'figure', allow_duplicate=True),
        [Input('selected-cluster-store', 'data'),
         Input('clusters-store', 'data')],
        [State('selected-feature-store', 'data')],
        prevent_initial_call=True
    )
    def handle_cluster_highlighting_only(selected_clusters, cluster_data, selected_feature):
        """Handle ONLY cluster highlighting - separate from node highlighting."""
        
        # Handle cluster highlighting updates
        
        # If a node is selected, include it in the cluster highlighting
        selected_feature_id = selected_feature if selected_feature is not None else None
        
        # Process cluster highlighting
        cluster_highlighted_nodes = set()
        if selected_clusters and cluster_data:
            # Ensure selected_clusters is a list
            if isinstance(selected_clusters, str):
                selected_clusters = [selected_clusters]
            elif not isinstance(selected_clusters, list):
                selected_clusters = []
            
            # Process selected clusters
            
            for cluster_id in selected_clusters:
                if cluster_id in cluster_data:
                    cluster_nodes = cluster_data[cluster_id].get('nodes', [])
                    # Add cluster nodes to highlighting
                    for node_data in cluster_nodes:
                        node_index = node_data.get('node_index')
                        if node_index is not None:
                            cluster_highlighted_nodes.add(node_index)
                # Skip missing clusters
        
        # Apply cluster highlighting to graph
        
        # Create cluster highlighting while preserving individual node selection
        figure = graph_component.create_graph_figure(
            selected_feature_id=selected_feature_id,  # Preserve individual node highlighting
            nodes_with_descriptions=set(),
            cluster_data=cluster_data,
            cluster_highlighted_nodes=cluster_highlighted_nodes
        )
        
        return figure

def handle_node_click_fast(click_data, current_selection, clicked_nodes_data, graph_component, viewport_data):
    """Fast path for node clicks without cluster interference."""
    if clicked_nodes_data is None:
        clicked_nodes_data = {
            'nodes_with_descriptions': [],
            'double_clicked_node': None,
            'last_click_time': 0,
            'last_clicked_node': None
        }
    
    # Handle viewport dimensions (only if they changed significantly)
    if viewport_data:
        width = viewport_data.get('width', 1600)
        height = viewport_data.get('height', 1000)
        
        # Only update dimensions if they changed significantly
        current_width = getattr(graph_component, 'screen_width', 0)
        current_height = getattr(graph_component, 'screen_height', 0)
        
        if (abs(width * 0.68 - current_width) > 50 or abs(height - 96 - current_height) > 50):
            graph_component.set_screen_dimensions(width, height)
    
    # Process click data
    double_clicked_node = clicked_nodes_data.get('double_clicked_node')
    if click_data and 'points' in click_data:
        point = click_data['points'][0]
        if 'pointIndex' in point:
            clicked_node_id = point['pointIndex']
            current_time = time.time()
            
            # Toggle highlighting
            if double_clicked_node == clicked_node_id:
                double_clicked_node = None
            else:
                double_clicked_node = clicked_node_id
            
            # Update click data
            clicked_nodes_data.update({
                'double_clicked_node': double_clicked_node,
                'nodes_with_descriptions': [],
                'last_click_time': current_time,
                'last_clicked_node': clicked_node_id
            })
    
    # Use ultra-fast highlighting (no clusters involved)
    figure = graph_component.create_graph_figure(
        selected_feature_id=double_clicked_node,
        nodes_with_descriptions=set(),
        cluster_data=None,
        cluster_highlighted_nodes=set()
    )
    
    # Create feature display
    display_content = create_feature_display(graph_component, double_clicked_node)
    
    return figure, display_content, double_clicked_node, clicked_nodes_data

def create_feature_display(graph_component: GraphComponent, 
                          selected_feature_id: Optional[int]):
    """Create the feature display content for double-clicked node."""
    
    # Create feature display content
    
    if selected_feature_id is None:
        # No feature selected
        return FeatureDisplay.create_activation_display(None, None)
    
    # Get the selected node
    selected_node = graph_component.get_feature_by_id(selected_feature_id)
    # Check if selected node exists
    
    if selected_node is None:
        # Selected node not found
        return FeatureDisplay.create_activation_display(None, None)
    
    # Get feature configuration
    feature_config = selected_node.config
    # Get feature configuration
    
    # Create feature info
    feature_info = {
        'layer': selected_node.layer,
        'pos': selected_node.pos,
        'feature_idx': selected_node.feature_idx,
        'token': selected_node.token,
        'description': selected_node.description
    }
    # Prepare feature info
    
    # Get intervention data for this specific feature node
    intervention_data = None
    graph_data = graph_component.get_graph_data()
    if graph_data.intervention_data:
        # Find the node with the selected feature id
        selected_node = None
        for node in graph_data.nodes:
            if node.id == selected_feature_id:
                selected_node = node
                break
        
        if selected_node:
            # Match intervention data by feature coordinates (layer, position, feature_idx)
            # instead of array index to fix mismatch between filtered nodes and unfiltered intervention data
            for interv_data in graph_data.intervention_data:
                if interv_data and interv_data.feature_info:
                    if (interv_data.feature_info.get('layer') == selected_node.layer and
                        interv_data.feature_info.get('position') == selected_node.pos and
                        interv_data.feature_info.get('feature_idx') == selected_node.feature_idx):
                        intervention_data = interv_data
                        break
    
    # Create activation display
    
    return FeatureDisplay.create_activation_display(feature_config, feature_info, intervention_data)
