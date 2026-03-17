import uuid
import math
import random
import os
import numpy as np
from dash import Input, Output, State, no_update, callback_context, html, ALL, dcc

from ..visualization.components.graph_component import GraphComponent
from ..visualization.components.cluster_manager import ClusterManager

def register_cluster_callbacks(app, graph_component: GraphComponent, cluster_manager: ClusterManager):
    """Simplified callbacks for cluster management without drag and drop."""
    
    # Add the auto-cluster controls callback
    @app.callback(
        Output('cluster-controls', 'style'),
        Input('auto-cluster-btn', 'n_clicks'),
        Input('cancel-cluster-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def toggle_cluster_controls(auto_clicks, cancel_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return {'display': 'none', 'backgroundColor': '#f8f9fa'}
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'auto-cluster-btn':
            return {'display': 'block', 'backgroundColor': '#f8f9fa'}
        else:  # cancel-cluster-btn
            return {'display': 'none', 'backgroundColor': '#f8f9fa'}
    
    # Add the create clusters callback
    @app.callback(
        [Output('clusters-store', 'data', allow_duplicate=True),  # Make sure this has allow_duplicate=True
         Output('cluster-controls', 'style', allow_duplicate=True)],
        Input('create-clusters-btn', 'n_clicks'),
        [State('cluster-count-dropdown', 'value'),
         State('cluster-method-dropdown', 'value'),
         State('clusters-store', 'data')],
        prevent_initial_call=True
    )
    def create_auto_clusters(n_clicks, n_clusters, use_activation_rate, existing_clusters):
        if not n_clicks:
            return no_update, no_update
        
        # Clear existing clusters
        clusters = {}
        
        # Get cluster assignments using the real clustering function
        cluster_labels = graph_component.data_loader.cluster_features(
            n_clusters=n_clusters, 
            use_activation_rate=use_activation_rate
        )
        
        if cluster_labels is None:
            return existing_clusters or {}, {'display': 'none', 'backgroundColor': '#f8f9fa'}
        
        # Get graph data to access feature information
        graph_data = graph_component.get_graph_data()
        
        # Get unique cluster IDs (excluding -1 for unclustered)
        unique_cluster_ids = np.unique(cluster_labels)
        unique_cluster_ids = unique_cluster_ids[unique_cluster_ids >= 0]  # Remove -1 (unclustered)
        
        # Calculate cluster positions in a grid layout
        cluster_positions = calculate_cluster_positions(len(unique_cluster_ids))
        
        # Create clusters based on assignments
        for cluster_id in unique_cluster_ids:
            cluster_uuid = str(uuid.uuid4())
            color = cluster_manager.cluster_colors[cluster_id % len(cluster_manager.cluster_colors)]
            
            # Find all features assigned to this cluster
            cluster_nodes = []
            for feature_idx, assigned_cluster in enumerate(cluster_labels):
                if assigned_cluster == cluster_id and feature_idx < len(graph_data.nodes):
                    node = graph_data.nodes[feature_idx]
                    cluster_nodes.append({
                        'node_index': feature_idx,
                        'layer': node.layer,
                        'pos': node.pos,
                        'feature_idx': node.feature_idx,
                        'description': getattr(node, 'description', ''),
                        'x': node.x,
                        'y': node.y,
                        'cluster_id': cluster_id,  # Add cluster ID for coloring
                        'cluster_color': color     # Add cluster color
                    })
            
            # Only create cluster if it has nodes
            if cluster_nodes:
                cluster_pos = cluster_positions[cluster_id]
                clusters[cluster_uuid] = {
                    'name': f'Cluster {cluster_id + 1}',
                    'color': color,
                    'nodes': cluster_nodes,
                    'x': cluster_pos['x'],
                    'y': cluster_pos['y'],
                    'cluster_id': cluster_id,  # Store cluster ID for graph coloring
                    'description': f'Group{cluster_id + 1}'  # Add default description
                }
        
        print(f"Created {len(clusters)} clusters from {n_clusters} requested clusters using {'activation rate' if use_activation_rate else 'normalized score'}")
        
        return clusters, {'display': 'none', 'backgroundColor': '#f8f9fa'}


    # Add callback to handle tab switching
    @app.callback(
        [Output('active-tab-store', 'data'),
         Output('manual-tab-btn', 'style'),
         Output('tsne-tab-btn', 'style'),
         Output('manual-tab-content', 'style'),
         Output('tsne-tab-content', 'style'),
         Output('tsne-plot', 'figure', allow_duplicate=True)],
        [Input('manual-tab-btn', 'n_clicks'),
         Input('tsne-tab-btn', 'n_clicks')],
        [State('active-tab-store', 'data'),
         State('cluster-method-dropdown', 'value'),
         State('embedding-data-store', 'data')],
        prevent_initial_call=True
    )
    def handle_tab_switch(manual_clicks, tsne_clicks, current_tab_data, use_activation_rate, embedding_data):
        """Handle switching between manual clustering and t-SNE tabs."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_tab = current_tab_data.get('tab', 'manual') if current_tab_data else 'manual'
        
        # Determine new tab
        if trigger_id == 'manual-tab-btn':
            new_tab = 'manual'
        elif trigger_id == 'tsne-tab-btn':
            new_tab = 'tsne'
        else:
            new_tab = current_tab
        
        # Don't do anything if clicking the already active tab
        if new_tab == current_tab:
            return no_update, no_update, no_update, no_update, no_update, no_update
        
        # Tab button styles - maintain consistent size and styling
        manual_style = {
            'backgroundColor': '#f0f9ff' if new_tab == 'manual' else '#f8fafc',
            'color': '#0284c7' if new_tab == 'manual' else '#64748b',
            'border': '1px solid #bae6fd' if new_tab == 'manual' else '1px solid #e2e8f0',
            'padding': '4px 10px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '10px',
            'fontWeight': '600',
            'marginRight': '2px',
            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            'minWidth': '50px',
            'textAlign': 'center'
        }
        
        tsne_style = {
            'backgroundColor': '#f0f9ff' if new_tab == 'tsne' else '#f8fafc',
            'color': '#0284c7' if new_tab == 'tsne' else '#64748b',
            'border': '1px solid #bae6fd' if new_tab == 'tsne' else '1px solid #e2e8f0',
            'padding': '4px 10px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '10px',
            'fontWeight': '600',
            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
            'minWidth': '50px',
            'textAlign': 'center'
        }
        
        # Tab content styles
        manual_content_style = {
            'display': 'block' if new_tab == 'manual' else 'none',
            'height': '100%',
            'backgroundColor': 'white'
        }
        
        tsne_content_style = {
            'display': 'block' if new_tab == 'tsne' else 'none',
            'height': '100%',
            'backgroundColor': 'white'
        }
        
        # Don't auto-compute t-SNE, let the on-demand callback handle it
        tsne_figure = {}
        
        return {'tab': new_tab}, manual_style, tsne_style, manual_content_style, tsne_content_style, tsne_figure

    # Add on-demand t-SNE computation callback
    @app.callback(
        [Output('embedding-data-store', 'data'),
         Output('tsne-plot', 'figure')],
        [Input('tsne-cluster-count-dropdown', 'value'),
         Input('tsne-cluster-method-dropdown', 'value')],
        [State('active-tab-store', 'data')],
        prevent_initial_call=True
    )
    def compute_tsne_on_demand(n_clusters, use_activation_rate, tab_state):
        """Compute t-SNE/UMAP on-demand when user changes settings."""
        # Only compute if we're on the t-SNE tab
        if not tab_state or tab_state.get('tab') != 'tsne':
            return {}, {}
        
        # PERFORMANCE: Check cache to avoid expensive recomputation
        cache_key = f"tsne_{use_activation_rate}"
        if hasattr(graph_component, '_tsne_cache') and cache_key in graph_component._tsne_cache:
            print(f"Using cached t-SNE for activation_rate={use_activation_rate}")
            cached_result = graph_component._tsne_cache[cache_key]
            return cached_result['embedding_data'], cached_result['cluster_data']
        
        print(f"Computing t-SNE with {n_clusters} clusters, activation_rate={use_activation_rate}")
        
        # Compute new embeddings
        embedding_results = graph_component.data_loader.compute_embeddings(
            use_activation_rate=use_activation_rate
        )
        
        if embedding_results is None:
            print(f"ERROR: No embedding results returned for activation_rate={use_activation_rate}")
            return {}, {}
        
        print(f"Embedding results keys: {embedding_results.keys()}")
        if 'tsne' in embedding_results:
            print(f"t-SNE shape: {np.array(embedding_results['tsne']).shape}")
        if 'umap' in embedding_results:
            print(f"UMAP shape: {np.array(embedding_results['umap']).shape}")
        
        # Store embedding data
        new_embedding_data = {
            'tsne': embedding_results.get('tsne', []).tolist() if 'tsne' in embedding_results else [],
            'umap': embedding_results.get('umap', []).tolist() if 'umap' in embedding_results else [],
            'method_used': embedding_results.get('method_used', 'activation_rate'),
            'active_indices': embedding_results.get('active_indices', []).tolist() if 'active_indices' in embedding_results else []
        }
        
        # Apply clustering if requested
        cluster_assignments = None
        if n_clusters > 0:
            cluster_labels = graph_component.data_loader.cluster_features(
                n_clusters=n_clusters, 
                use_activation_rate=use_activation_rate
            )
            if cluster_labels is not None:
                cluster_assignments = {'labels': cluster_labels}
        
        # Create t-SNE plot with optional clustering
        tsne_figure = create_tsne_plot(new_embedding_data, cluster_assignments)
        
        return new_embedding_data, tsne_figure

    # Add callback to handle t-SNE clustering
    @app.callback(
        [Output('tsne-clusters-store', 'data'),
         Output('tsne-plot', 'figure', allow_duplicate=True)],
        [Input('tsne-cluster-count-dropdown', 'value'),
         Input('tsne-cluster-method-dropdown', 'value')],
        [State('embedding-data-store', 'data'),
         State('active-tab-store', 'data')],
        prevent_initial_call=True
    )
    def handle_tsne_clustering(n_clusters, use_activation_rate, embedding_data, tab_data):
        """Handle clustering of t-SNE points and update plot colors."""
        
        # Only process if we're on the t-SNE tab
        if not tab_data or tab_data.get('tab') != 'tsne':
            return no_update, no_update
        
        if not embedding_data or not (embedding_data.get('tsne') or embedding_data.get('umap')):
            return no_update, no_update
        
        cluster_assignments = {}
        
        if n_clusters > 0:
            # Perform clustering on the original correlation matrix
            cluster_labels = graph_component.data_loader.cluster_features(
                n_clusters=n_clusters, 
                use_activation_rate=use_activation_rate
            )
            
            if cluster_labels is not None:
                cluster_assignments = {
                    'labels': cluster_labels.tolist(),
                    'n_clusters': n_clusters,
                    'method': 'activation_rate' if use_activation_rate else 'normalized_score'
                }
        
        # Create updated t-SNE plot with cluster colors
        tsne_figure = create_tsne_plot(embedding_data, cluster_assignments)
        
        return cluster_assignments, tsne_figure

    # Add callback to handle t-SNE plot clicks
    @app.callback(
        Output('selected-feature-store', 'data', allow_duplicate=True),
        Input('tsne-plot', 'clickData'),
        [State('tsne-toggle-store', 'data'),
         State('embedding-data-store', 'data')],
        prevent_initial_call=True
    )
    def handle_tsne_click(click_data, tsne_state, embedding_data):
        """Handle clicks on t-SNE plot points to select features in main graph."""
        
        if not click_data or 'points' not in click_data:
            return no_update
        
        if not embedding_data or 'active_indices' not in embedding_data:
            return no_update
        
        # Get the clicked point
        point = click_data['points'][0]
        
        # Handle different trace structures (clustered vs non-clustered)
        original_feature_index = None
        clicked_point_index = None
        
        # First try to get the point index from customdata (for multi-trace plots)
        if 'customdata' in point and point['customdata'] is not None:
            if isinstance(point['customdata'], list) and len(point['customdata']) > 0:
                clicked_point_index = point['customdata'][0]
            else:
                clicked_point_index = point['customdata']
        
        # If no customdata, use pointIndex within the trace
        if clicked_point_index is None and 'pointIndex' in point:
            # For multi-trace plots, we need to accumulate point indices from previous traces
            point_in_trace = point['pointIndex']
            
            # Calculate the global point index by adding points from previous traces
            clicked_point_index = point_in_trace
            
            # If this is a multi-trace plot (clustered), we need to find the original point index
            # by looking at the customdata of the clicked point
            if 'customdata' in point and point['customdata'] is not None:
                if isinstance(point['customdata'], list) and len(point['customdata']) > 0:
                    clicked_point_index = point['customdata'][0]
                else:
                    clicked_point_index = point['customdata']
        
        if clicked_point_index is None:
            return no_update
        
        # Map from t-SNE point index to original feature index
        active_indices = embedding_data['active_indices']
        if clicked_point_index < len(active_indices):
            original_feature_index = active_indices[clicked_point_index]
            return original_feature_index
        
        return no_update

    def calculate_cluster_positions(n_clusters):
        """Calculate positions for clusters in a grid layout."""
        # Calculate grid dimensions
        cols = math.ceil(math.sqrt(n_clusters))
        rows = math.ceil(n_clusters / cols)
        
        # Cluster area dimensions (adjust these based on your cluster graph size)
        cluster_width = 150
        cluster_height = 100
        margin = 50
        
        # Calculate starting positions
        total_width = cols * cluster_width + (cols - 1) * margin
        total_height = rows * cluster_height + (rows - 1) * margin
        start_x = -total_width / 2
        start_y = -total_height / 2
        
        positions = []
        for i in range(n_clusters):
            col = i % cols
            row = i // cols
            
            x = start_x + col * (cluster_width + margin)
            y = start_y + row * (cluster_height + margin)
            
            positions.append({'x': x, 'y': y})
        
        return positions

    @app.callback(
        Output('clusters-store', 'data', allow_duplicate=True),  # Make sure this has allow_duplicate=True
        Input('add-cluster-btn', 'n_clicks'),
        State('clusters-store', 'data'),
        prevent_initial_call=True
    )
    def add_cluster(n_clicks, clusters):
        if not n_clicks:
            return clusters or {}
        clusters = clusters or {}
        cid = str(uuid.uuid4())
        color = cluster_manager.cluster_colors[len(clusters) % len(cluster_manager.cluster_colors)]
        clusters[cid] = {
            'name': f'Cluster {len(clusters) + 1}',
            'color': color,
            'nodes': [],
            'description': 'cluster'  # Add default description
        }
        return clusters

    @app.callback(
        [Output('link-mode-store', 'data'),
         Output('link-mode-btn', 'style'),
         Output('delete-mode-store', 'data'),
         Output('delete-mode-btn', 'style'),
         Output('intervention-mode-store', 'data'),
         Output('intervention-mode-btn', 'style'),
         Output('mode-indicator', 'children'),
         Output('mode-indicator', 'style')],
        [Input('link-mode-btn', 'n_clicks'),
         Input('delete-mode-btn', 'n_clicks'),
         Input('intervention-mode-btn', 'n_clicks')],
        [State('link-mode-store', 'data'),
         State('delete-mode-store', 'data'),
         State('intervention-mode-store', 'data')],
        prevent_initial_call=True
    )
    def toggle_modes(link_clicks, delete_clicks, intervention_clicks, link_mode_data, delete_mode_data, intervention_mode_data):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Default states
        link_active = link_mode_data.get('active', False) if link_mode_data else False
        delete_active = delete_mode_data.get('active', False) if delete_mode_data else False
        intervention_active = intervention_mode_data.get('active', False) if intervention_mode_data else False
        
        if trigger_id == 'link-mode-btn':
            link_active = not link_active
            delete_active = False
            intervention_active = False
        elif trigger_id == 'delete-mode-btn':
            delete_active = not delete_active
            link_active = False
            intervention_active = False
        elif trigger_id == 'intervention-mode-btn':
            intervention_active = not intervention_active
            link_active = False
            delete_active = False
        
        # Button styles
        link_style = {
            'backgroundColor': '#3b82f6' if link_active else '#ffffff',
            'color': 'white' if link_active else '#6b7280',
            'border': '1px solid #e5e7eb',
            'padding': '6px 12px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '11px',
            'fontWeight': '400',
            'transition': 'all 0.2s ease',
            'boxShadow': 'none',
            'marginRight': '6px'
        }
        
        delete_style = {
            'backgroundColor': '#ef4444' if delete_active else '#ffffff',
            'color': 'white' if delete_active else '#6b7280',
            'border': '1px solid #e5e7eb',
            'padding': '6px 12px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '11px',
            'fontWeight': '400',
            'transition': 'all 0.2s ease',
            'boxShadow': 'none',
            'marginRight': '4px'
        }
        
        intervention_style = {
            'backgroundColor': '#ca8a04' if intervention_active else '#fefce8',
            'color': 'white' if intervention_active else '#ca8a04',
            'border': '1px solid #fed7aa',
            'padding': '6px 12px',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '11px',
            'fontWeight': '400',
            'transition': 'all 0.2s ease',
            'boxShadow': 'none'
        }
        
        # Mode indicator
        if link_active:
            indicator_text = "Link Mode: Click two clusters to connect them"
            indicator_style = {
                'padding': '6px 16px',
                'fontSize': '10px',
                'backgroundColor': '#dbeafe',
                'color': '#1e40af',
                'borderBottom': '1px solid #f0f0f0',
                'display': 'block'
            }
        elif delete_active:
            indicator_text = "Delete Mode: Click clusters to remove them"
            indicator_style = {
                'padding': '6px 16px',
                'fontSize': '10px',
                'backgroundColor': '#fee2e2',
                'color': '#dc2626',
                'borderBottom': '1px solid #f0f0f0',
                'display': 'block'
            }
        elif intervention_active:
            indicator_text = "Intervention Mode: Click a cluster to run intervention"
            indicator_style = {
                'padding': '6px 16px',
                'fontSize': '10px',
                'backgroundColor': '#fefce8',
                'color': '#ca8a04',
                'borderBottom': '1px solid #f0f0f0',
                'display': 'block'
            }
        else:
            indicator_text = ""
            indicator_style = {
                'padding': '6px 16px',
                'fontSize': '10px',
                'backgroundColor': '#f8f9fa',
                'color': '#9ca3af',
                'borderBottom': '1px solid #f0f0f0',
                'display': 'none'
            }
        
        return (
            {'active': link_active, 'first_cluster': None},
            link_style,
            {'active': delete_active},
            delete_style,
            {'active': intervention_active, 'selected_cluster': None},
            intervention_style,
            indicator_text,
            indicator_style
        )
    
    @app.callback(
        [Output('clusters-store', 'data', allow_duplicate=True),
         Output('manual-edges-store', 'data'),
         Output('cluster-positions-store', 'data', allow_duplicate=True),
         Output('delete-mode-store', 'data', allow_duplicate=True),
         Output('link-mode-store', 'data', allow_duplicate=True),
         Output('intervention-mode-store', 'data', allow_duplicate=True),
         Output('intervention-dialog', 'style'),
         Output('selected-feature-store', 'data', allow_duplicate=True)],
        Input('cluster-graph', 'tapNodeData'),
        [State('clusters-store', 'data'),
         State('manual-edges-store', 'data'),
         State('cluster-positions-store', 'data'),
         State('delete-mode-store', 'data'),
         State('link-mode-store', 'data'),
         State('intervention-mode-store', 'data'),
         State('selected-feature-store', 'data')],
        prevent_initial_call=True
    )
    def handle_cluster_interactions(node_data, clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, selected_feature):
        """Handle cluster interactions: delete, link, add features, intervention."""
        
        if not node_data or not clusters:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Get cluster ID
        cid = node_data.get('id')
        print(f"Cluster interaction: {cid}")
        
        # Handle feature nodes (extract cluster ID from feature node ID)
        if '_feature_' in cid:
            cid = cid.split('_feature_')[0]
        
        # Ignore label nodes - they'll be handled by the text input system
        if cid.endswith('_label'):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Skip if this is just a selection click (not in any mode and no selected feature)
        if not delete_mode_data.get('active', False) and not link_mode_data.get('active', False) and not intervention_mode_data.get('active', False) and not selected_feature:
            print("Regular cluster click - handled by selection callback")
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        if cid not in clusters:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Initialize defaults
        manual_edges = manual_edges or []
        positions = positions or {}
        delete_mode_data = delete_mode_data or {'active': False}
        link_mode_data = link_mode_data or {'active': False, 'first_cluster': None}
        intervention_mode_data = intervention_mode_data or {'active': False, 'selected_cluster': None}
        
        # Hidden dialog style
        hidden_dialog_style = {
            'position': 'fixed',
            'display': 'none',
            'zIndex': 1001,
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)'
        }
        
        # Handle delete mode
        if delete_mode_data.get('active', False):
            print(f"Deleting cluster: {cid}")
            
            # Remove cluster from clusters store
            new_clusters = {k: v for k, v in clusters.items() if k != cid}
            
            # Remove any edges connected to this cluster
            new_edges = [
                edge for edge in manual_edges 
                if edge['source'] != cid and edge['target'] != cid
            ]
            
            # Remove cluster position
            new_positions = {k: v for k, v in positions.items() if k != cid}
            
            # Turn off delete mode after deletion
            new_delete_mode = {'active': False}
            
            return new_clusters, new_edges, new_positions, new_delete_mode, link_mode_data, intervention_mode_data, hidden_dialog_style, selected_feature
        
        # Handle link mode
        elif link_mode_data.get('active', False):
            first_cluster = link_mode_data.get('first_cluster')
            
            if first_cluster is None:
                # First cluster selected
                print(f"First cluster selected for linking: {cid}")
                return clusters, manual_edges, positions, delete_mode_data, {
                    'active': True,
                    'first_cluster': cid
                }, intervention_mode_data, hidden_dialog_style, selected_feature
            elif first_cluster == cid:
                # Same cluster clicked twice, deselect
                print(f"Deselecting cluster for linking: {cid}")
                return clusters, manual_edges, positions, delete_mode_data, {
                    'active': True,
                    'first_cluster': None
                }, intervention_mode_data, hidden_dialog_style, selected_feature
            else:
                # Second cluster selected, create edge
                print(f"Creating link between {first_cluster} and {cid}")
                new_edge = {
                    'source': first_cluster,
                    'target': cid,
                    'id': f"{first_cluster}_{cid}"
                }
                
                # Check if edge already exists
                edge_exists = any(
                    (edge['source'] == first_cluster and edge['target'] == cid) or
                    (edge['source'] == cid and edge['target'] == first_cluster)
                    for edge in manual_edges
                )
                
                if not edge_exists:
                    manual_edges = manual_edges + [new_edge]
                
                # Reset link mode
                return clusters, manual_edges, positions, delete_mode_data, {
                    'active': False,
                    'first_cluster': None
                }, intervention_mode_data, hidden_dialog_style, selected_feature
        
        # Handle adding single selected feature
        elif selected_feature:
            print(f"Adding selected feature to cluster: {cid}")
            # Get node index
            node_index = None
            if isinstance(selected_feature, dict):
                node_index = selected_feature.get('node_index')
            elif isinstance(selected_feature, int):
                node_index = selected_feature
            
            if node_index is None:
                return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, selected_feature

            # Get node data efficiently - check bounds first to avoid expensive call
            try:
                graph_data = graph_component.get_graph_data()
                if node_index >= len(graph_data.nodes):
                    return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, selected_feature
                node = graph_data.nodes[node_index]
            except (IndexError, AttributeError):
                return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, selected_feature
            
            # Check if already in cluster
            existing = clusters[cid].get('nodes', [])
            if any(n['node_index'] == node_index for n in existing):
                return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, None  # Clear selected feature

            # Add to cluster
            entry = {
                'node_index': node_index,
                'layer': node.layer,
                'pos': node.pos,
                'feature_idx': node.feature_idx,
                'description': getattr(node, 'description', ''),
                'x': node.x,
                'y': node.y
            }
            
            clusters = dict(clusters)
            clusters[cid]['nodes'] = existing + [entry]
            # Set default description if not set
            if 'description' not in clusters[cid]:
                clusters[cid]['description'] = 'cluster'
            
            # Clear the selected feature after adding it
            return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, None
        
        # Handle intervention mode
        elif intervention_mode_data.get('active', False):
            print(f"Cluster clicked in intervention mode: {cid}")
            
            # Show intervention dialog
            show_dialog_style = {
                'position': 'fixed',
                'display': 'block',
                'zIndex': 1001,
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)'
            }
            
            # Update intervention mode to store selected cluster
            new_intervention_mode = {
                'active': True,
                'selected_cluster': cid
            }
            
            return clusters, manual_edges, positions, delete_mode_data, link_mode_data, new_intervention_mode, show_dialog_style, selected_feature
        
        return clusters, manual_edges, positions, delete_mode_data, link_mode_data, intervention_mode_data, hidden_dialog_style, selected_feature

    @app.callback(
        Output('selected-cluster-store', 'data'),
        Input('cluster-graph', 'tapNodeData'),
        [State('selected-cluster-store', 'data'),
         State('clusters-store', 'data'),
         State('delete-mode-store', 'data'),
         State('link-mode-store', 'data'),
         State('selected-feature-store', 'data')],
        prevent_initial_call=True
    )
    def handle_cluster_selection(node_data, selected_clusters, clusters, delete_mode_data, link_mode_data, selected_feature):
        """Handle cluster selection for highlighting nodes on main graph."""
        
        print("=== CLUSTER SELECTION CALLBACK ===")
        print(f"Input selected_clusters: {selected_clusters} (type: {type(selected_clusters)})")
        print(f"node_data: {node_data}")
        print(f"selected_feature: {selected_feature}")
        
        if not node_data or not clusters:
            print("No node_data or clusters, returning current selection")
            return selected_clusters or []
        
        # Get cluster ID
        cid = node_data.get('id')
        print(f"Cluster tap detected: {cid}")
        
        # Handle feature nodes (extract cluster ID from feature node ID)
        if '_feature_' in cid:
            cid = cid.split('_feature_')[0]
        
        # Ignore label nodes for selection
        if cid.endswith('_label'):
            print("Ignoring label node")
            return selected_clusters or []
        
        # DON'T handle selection if there's a selected feature (let interaction callback handle it)
        if selected_feature:
            print(f"Selected feature exists ({selected_feature}), skipping cluster selection")
            return selected_clusters or []
        
        # Only handle selection if not in delete or link mode
        if (delete_mode_data and delete_mode_data.get('active', False)) or \
           (link_mode_data and link_mode_data.get('active', False)):
            print("Ignoring cluster selection - in delete/link mode")
            return selected_clusters or []
        
        # Initialize selected_clusters as list if None
        if selected_clusters is None:
            selected_clusters = []
        elif not isinstance(selected_clusters, list):
            print(f"Converting selected_clusters from {type(selected_clusters)} to list")
            selected_clusters = []
        
        print(f"Current selected_clusters: {selected_clusters}")
        
        # Check if this is a valid cluster
        if cid in clusters:
            # SINGLE SELECTION: clicking cluster selects only that cluster
            # If same cluster clicked again, deselect it
            if selected_clusters and cid in selected_clusters:
                # Deselect current cluster
                new_selected_clusters = []
                print(f"Deselected cluster {cid}")
            else:
                # Select only this cluster (clear any previous selection)
                new_selected_clusters = [cid]
                print(f"Selected cluster {cid} (cleared previous selection)")
            
            print(f"Returning: {new_selected_clusters}")
            return new_selected_clusters
        else:
            print(f"Cluster {cid} not found in clusters")
        
        print(f"No change, returning: {selected_clusters}")
        return selected_clusters or []

    # Clear node highlighting when cluster is selected
    @app.callback(
        [Output('selected-feature-store', 'data', allow_duplicate=True),
         Output('clicked-nodes-store', 'data', allow_duplicate=True)],
        Input('selected-cluster-store', 'data'),
        prevent_initial_call=True
    )
    def clear_node_highlighting_on_cluster_select(selected_clusters):
        """Clear node highlighting when clusters are selected."""
        if selected_clusters and len(selected_clusters) > 0:
            # Clear any existing node selection/highlighting
            return None, {
                'nodes_with_descriptions': [],
                'double_clicked_node': None,
                'last_click_time': 0,
                'last_clicked_node': None
            }
        return no_update, no_update

    @app.callback(
        Output('cluster-graph', 'elements'),
        [Input('clusters-store', 'data'),
         Input('manual-edges-store', 'data'),
         Input('link-mode-store', 'data'),
         Input('delete-mode-store', 'data')],
        State('cluster-positions-store', 'data'),
        prevent_initial_call=True
    )
    def update_cytoscape_elements(clusters, manual_edges, link_mode_data, delete_mode_data, positions):
        if not clusters:
            return []
        
        elements = []
        first_cluster = link_mode_data.get('first_cluster') if link_mode_data else None
        delete_active = delete_mode_data.get('active', False) if delete_mode_data else False
        
        # Only get graph_data when we actually need it for connections
        graph_data = None
        
        # Add cluster nodes and their feature nodes
        for cid, data in clusters.items():
            cluster_nodes = data.get('nodes', [])
            cluster_color = data['color']
            # cluster_name = data['name']
            # cluster_description = data.get('description', 'Cluster')  # Get description or default
            
            # Get cluster position from stored positions or use default
            if positions and cid in positions:
                pos = positions[cid]
            else:
                pos = {'x': data.get('x', 0), 'y': data.get('y', 0)}
            
            # Determine cluster class
            cluster_class = 'cluster-empty' if not cluster_nodes else 'cluster-filled'
            if cid == first_cluster:
                cluster_class += ' cluster-highlight'
            elif delete_active:
                cluster_class += ' cluster-delete'
            
            if not cluster_nodes:
                # Empty cluster - show + for add mode
                label = '×' if delete_active else '+'
                elements.append({
                    'data': {
                        'id': cid,
                        'label': label,
                        'bgcolor': cluster_color
                    },
                    'position': pos,
                    'classes': cluster_class
                })
            else:
                # Filled cluster
                cluster_width, cluster_height = _calculate_cluster_size(cluster_nodes)
                
                elements.append({
                    'data': {
                        'id': cid,
                        'label': '',
                        'width': cluster_width,
                        'height': cluster_height
                    },
                    'position': pos,
                    'classes': cluster_class
                })
                
                # Add cluster label at center bottom (absolute positioning)
                cluster_description = data.get('description', 'cluster')
                elements.append({
                    'data': {
                        'id': f'{cid}_label',
                        'description': cluster_description
                    },
                    'position': {'x': pos['x'], 'y': pos['y'] + cluster_height/2 + 15},  # Position slightly below cluster
                    'classes': 'cluster-label',
                    'grabbable': False,  # Prevent dragging the label
                    'selectable': True   # Keep it selectable for clicking
                })
                
                # Add feature nodes
                feature_positions = _calculate_feature_positions_horizontal(cluster_nodes, pos, cluster_width, cluster_height)
                for i, (node_data, feature_pos) in enumerate(zip(cluster_nodes, feature_positions)):
                    feature_id = f"{cid}_feature_{i}"
                    
                    elements.append({
                        'data': {
                            'id': feature_id,
                            'label': '',
                            'bgcolor': cluster_color,
                            'parent': cid
                        },
                        'position': feature_pos,
                        'classes': 'feature-node'
                    })
        
        # Add manual edges
        for edge in manual_edges:
            elements.append({
                'data': {
                    'id': edge['id'],
                    'source': edge['source'],
                    'target': edge['target']
                },
                'classes': 'cluster-edge-manual'
            })
        
        # Add automatic edges (based on feature connections)
        cluster_keys = list(clusters.keys())
        for i, cid1 in enumerate(cluster_keys):
            for cid2 in cluster_keys[i+1:]:
                if clusters[cid1]['nodes'] and clusters[cid2]['nodes']:
                    # Check if manual edge already exists
                    manual_edge_exists = any(
                        (edge['source'] == cid1 and edge['target'] == cid2) or
                        (edge['source'] == cid2 and edge['target'] == cid1)
                        for edge in manual_edges
                    )
                    
                    if not manual_edge_exists:
                        # Lazy load graph_data only when needed for connections
                        if graph_data is None:
                            graph_data = graph_component.get_graph_data()
                        
                        # Skip auto-connections computation for performance when there are many nodes
                        # Auto-connections are nice-to-have but not essential for functionality
                        total_cluster_nodes = len(clusters[cid1]['nodes']) + len(clusters[cid2]['nodes'])
                        if total_cluster_nodes > 20:  # Skip if clusters are large
                            continue
                            
                        # Calculate connection strength
                        strength = _calculate_connection_strength(
                            clusters[cid1]['nodes'], clusters[cid2]['nodes'], graph_data
                        )
                        
                        if strength > 0.05:  # Only show strong connections
                            elements.append({
                                'data': {
                                    'id': f"auto_{cid1}_{cid2}",
                                    'source': cid1,
                                    'target': cid2,
                                    'weight': 1
                                },
                                'classes': 'cluster-edge-auto'
                            })
        
        return elements

    @app.callback(
        Output('cluster-positions-store', 'data', allow_duplicate=True),
        Input('cluster-graph', 'elements'),
        State('cluster-positions-store', 'data'),
        prevent_initial_call=True
    )
    def capture_positions(elements, old_positions):
        if not elements:
            return no_update
        
        newpos = old_positions.copy() if old_positions else {}
        
        for el in elements:
            if ('position' in el and 
                el['data'].get('id') and 
                not el['data']['id'].startswith('_') and
                '_feature_' not in el['data']['id'] and
                not el['data']['id'].startswith('auto_')):
                
                cluster_id = el['data']['id']
                newpos[cluster_id] = el['position']
        
        return newpos

    # Handle showing text input when clicking cluster labels
    @app.callback(
        [Output('cluster-label-input', 'style'),
         Output('cluster-label-input', 'value'),
         Output('editing-cluster-id', 'data')],
        Input('cluster-graph', 'tapNodeData'),
        [State('clusters-store', 'data'),
         State('editing-cluster-id', 'data')],
        prevent_initial_call=True
    )
    def show_text_input(node_data, clusters, editing_cluster_id):
        """Show text input when clicking cluster labels."""
        # Processing text input request
        
        if not node_data:
            # No node data provided
            return no_update, no_update, no_update
            
        cid = node_data.get('id', '')
        # Processing node click
        
        if cid.endswith('_label'):
            cluster_id = cid.replace('_label', '')
            # Editing cluster label
            if cluster_id in clusters:
                # Show the text input
                input_style = {
                    'position': 'fixed',
                    'display': 'block',
                    'zIndex': 1000,
                    'fontSize': '11px',
                    'padding': '4px 6px',
                    'border': '1px solid #3b82f6',
                    'borderRadius': '4px',
                    'outline': 'none',
                    'backgroundColor': 'white',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                    'width': '120px',
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)'
                }
                current_desc = clusters[cluster_id].get('description', '')
                # Opening text input
                return input_style, current_desc, cluster_id
            else:
                # Cluster not found
                pass
        else:
            # Not a label node
            pass
        
        return no_update, no_update, no_update


    # Handle saving text input when user presses Enter
    @app.callback(
        [Output('clusters-store', 'data', allow_duplicate=True),
         Output('cluster-label-input', 'style', allow_duplicate=True),
         Output('editing-cluster-id', 'data', allow_duplicate=True)],
        [Input('cluster-label-input', 'n_submit')],
        [State('cluster-label-input', 'value'),
         State('editing-cluster-id', 'data'),
         State('clusters-store', 'data')],
        prevent_initial_call=True
    )
    def save_text_input(n_submit, input_value, editing_cluster_id, clusters):
        """Save text input when user presses Enter."""
        if not n_submit or not editing_cluster_id or not clusters:
            return no_update, no_update, no_update
            
        # Update cluster description
        clusters = dict(clusters)
        clusters[editing_cluster_id]['description'] = input_value or 'cluster'
        
        # Hide the text input
        hidden_style = {
            'position': 'fixed',
            'display': 'none',
            'zIndex': 1000,
            'fontSize': '11px',
            'padding': '4px 6px',
            'border': '1px solid #3b82f6',
            'borderRadius': '4px',
            'outline': 'none',
            'backgroundColor': 'white',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
            'width': '120px'
        }
        
        return clusters, hidden_style, None

    # Handle closing text input when user presses Escape or clicks elsewhere
    @app.callback(
        [Output('cluster-label-input', 'style', allow_duplicate=True),
         Output('editing-cluster-id', 'data', allow_duplicate=True)],
        [Input('cluster-label-input', 'n_blur')],  # When input loses focus
        prevent_initial_call=True
    )
    def close_text_input_on_blur(n_blur):
        """Close text input when it loses focus."""
        if n_blur:
            hidden_style = {
                'position': 'fixed',
                'display': 'none',
                'zIndex': 1000,
                'fontSize': '11px',
                'padding': '4px 6px',
                'border': '1px solid #3b82f6',
                'borderRadius': '4px',
                'outline': 'none',
                'backgroundColor': 'white',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.15)',
                'width': '120px'
            }
            return hidden_style, None
        return no_update, no_update

    # Update label positions when clusters move
    @app.callback(
        Output('cluster-graph', 'elements', allow_duplicate=True),
        Input('cluster-graph', 'elements'),
        State('clusters-store', 'data'),
        prevent_initial_call=True
    )
    def update_label_positions(elements, clusters):
        """Keep labels positioned under their clusters when clusters move."""
        if not elements or not clusters:
            return no_update
        
        # Find cluster positions
        cluster_positions = {}
        updated_elements = []
        
        for element in elements:
            if 'position' in element:
                node_id = element['data']['id']
                if not node_id.endswith('_label') and '_feature_' not in node_id and not node_id.startswith('auto_'):
                    cluster_positions[node_id] = element['position']
        
        # Update elements with corrected label positions
        for element in elements:
            if 'position' in element:
                node_id = element['data']['id']
                if node_id.endswith('_label'):
                    cluster_id = node_id.replace('_label', '')
                    if cluster_id in cluster_positions and cluster_id in clusters:
                        cluster_pos = cluster_positions[cluster_id]
                        cluster_nodes = clusters[cluster_id].get('nodes', [])
                        cluster_height = _calculate_cluster_size(cluster_nodes)[1]
                        # Update label position to be under cluster
                        element = dict(element)
                        element['position'] = {
                            'x': cluster_pos['x'],
                            'y': cluster_pos['y'] + cluster_height/2 + 15
                        }
            updated_elements.append(element)
        
        return updated_elements

    # Handle intervention dialog buttons
    @app.callback(
        [Output('intervention-mode-store', 'data', allow_duplicate=True),
         Output('intervention-dialog', 'style', allow_duplicate=True),
         Output('intervention-results-store', 'data'),
         Output('intervention-progress-dialog', 'style'),
         Output('intervention-progress-content', 'children')],
        [Input('run-intervention-btn', 'n_clicks'),
         Input('cancel-intervention-btn', 'n_clicks')],
        [State('intervention-mode-store', 'data'),
         State('clusters-store', 'data'),
         State('cluster-intervention-strength', 'value'),
         State('manual-features-store', 'data'),
         State('freeze-attention-toggle', 'value')],
        prevent_initial_call=True
    )
    def handle_intervention_dialog(run_clicks, cancel_clicks, intervention_mode_data, clusters, cluster_intervention_strength, manual_features, freeze_attention_toggle):
        """Handle intervention dialog button clicks."""
        import subprocess
        import json
        from dash import callback_context
        
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Hidden dialog styles
        hidden_dialog_style = {
            'position': 'fixed',
            'display': 'none',
            'zIndex': 1001,
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)'
        }
        
        hidden_progress_style = {
            'position': 'fixed',
            'display': 'none',
            'zIndex': 1003,
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)'
        }
        
        # Reset intervention mode
        reset_intervention_mode = {
            'active': False,
            'selected_cluster': None
        }
        
        if trigger_id == 'cancel-intervention-btn':
            print("Intervention canceled")
            return reset_intervention_mode, hidden_dialog_style, no_update, hidden_progress_style, []
        
        elif trigger_id == 'run-intervention-btn':
            if not intervention_mode_data or not intervention_mode_data.get('selected_cluster'):
                print("No cluster selected for intervention")
                return reset_intervention_mode, hidden_dialog_style, no_update, hidden_progress_style, []
            
            selected_cluster_id = intervention_mode_data['selected_cluster']
            if selected_cluster_id not in clusters:
                print(f"Cluster {selected_cluster_id} not found")
                return reset_intervention_mode, hidden_dialog_style, no_update, hidden_progress_style, []
            
            cluster = clusters[selected_cluster_id]
            cluster_nodes = cluster.get('nodes', [])
            
            if not cluster_nodes:
                print("No features in selected cluster")
                return reset_intervention_mode, hidden_dialog_style, {'error': 'No features in selected cluster'}, hidden_progress_style, []
            
            # Convert cluster nodes to features list format
            cluster_features = []
            for node in cluster_nodes:
                feature_tuple = (
                    node.get('pos', 0),
                    node.get('layer', 0),
                    node.get('feature_idx', 0)
                )
                cluster_features.append(feature_tuple)
            
            print(f"Running intervention on {len(cluster_features)} features from cluster {cluster.get('name', selected_cluster_id)}")
            
            cluster_intervention_value = cluster_intervention_strength if cluster_intervention_strength is not None else -10.0
            
            # Get freeze_attention value from toggle (list contains 'freeze' if checked, empty if unchecked)
            # Default to True if the toggle component hasn't been initialized yet
            if freeze_attention_toggle is None:
                freeze_attention = True  # Default to freezing attention
            else:
                freeze_attention = 'freeze' in freeze_attention_toggle
            
            # Prepare manual features list
            manual_features_list = []
            if manual_features:
                for feature in manual_features:
                    manual_features_list.append((
                        feature.get('pos', 0),
                        feature.get('layer', 0),
                        feature.get('feature_idx', 0),
                        feature.get('intervention_value', -10.0)
                    ))
            
            # Use the same input string that was used to generate the original data
            graph_data = graph_component.get_graph_data()
            input_string = graph_data.input_str
            
            try:
                # Submit the Condor job using the .sub file
                submit_script = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/run_cluster_intervention.sub"
                
                # Write arguments to a file in the run_cluster folder
                temp_args_file = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/cluster_intervention_args.json"
                args_data = {
                    "cluster_features": cluster_features,
                    "manual_features": manual_features_list,
                    "input_string": input_string,
                    "cluster_intervention_value": cluster_intervention_value,
                    "freeze_attention": freeze_attention
                }
                
                with open(temp_args_file, 'w') as f:
                    json.dump(args_data, f)
                
                # Remove any existing results file before submitting new job
                results_file_path = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/cluster_intervention_results.json"
                if os.path.exists(results_file_path):
                    try:
                        import time
                        os.remove(results_file_path)
                        print(f"DEBUG: Removed existing results file at {time.strftime('%H:%M:%S')}: {results_file_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove existing results file: {e}")
                else:
                    import time
                    print(f"DEBUG: No existing results file to remove at {time.strftime('%H:%M:%S')}")
                
                # Pass only the temp file path as argument
                result = subprocess.run([
                    "condor_submit_bid", "2000",  # Bid amount
                    "-append", f'arguments = "{temp_args_file}"',
                    submit_script
                ], capture_output=True, text=True, cwd="/lustre/home/fdraye/projects/featflow")
                
                if result.returncode == 0:
                    print("Condor job submitted successfully")
                    print(f"Submit output: {result.stdout}")
                    
                    # Extract job ID from submit output
                    job_id = None
                    print(f"DEBUG: Parsing condor submit output: {result.stdout}")
                    for line in result.stdout.split('\n'):
                        print(f"DEBUG: Checking line: '{line}'")
                        if "submitted to cluster" in line:
                            try:
                                job_id = line.split()[-1].rstrip('.')
                                print(f"DEBUG: Extracted job ID: {job_id}")
                                break
                            except (IndexError, AttributeError):
                                print(f"DEBUG: Failed to extract job ID from line: '{line}'")
                                pass
                    
                    # Show progress dialog with job status
                    show_progress_style = {
                        'position': 'fixed',
                        'display': 'block',
                        'zIndex': 1003,
                        'top': '50%',
                        'left': '50%',
                        'transform': 'translate(-50%, -50%)'
                    }
                    
                    # Initial progress content
                    progress_content = [
                        html.Div([
                            html.Strong("🚀 Intervention Job Submitted", style={
                                'color': '#059669',
                                'fontSize': '14px',
                                'marginBottom': '10px',
                                'display': 'block'
                            }),
                            html.Div(f"Cluster: {cluster.get('description', 'cluster')}", style={
                                'fontSize': '12px',
                                'color': '#6b7280',
                                'marginBottom': '4px'
                            }),
                            html.Div(f"Features: {len(cluster_features)}", style={
                                'fontSize': '12px',
                                'color': '#6b7280',
                                'marginBottom': '4px'
                            }),
                            html.Div(f"Job ID: {job_id or 'N/A'}", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'fontFamily': 'monospace',
                                'marginBottom': '10px'
                            }),
                            html.Div("Checking job status...", style={
                                'fontSize': '11px',
                                'color': '#059669',
                                'fontStyle': 'italic'
                            })
                        ])
                    ]
                    
                    # Store job info for monitoring
                    job_info = {
                        'job_id': job_id,
                        'cluster_name': cluster.get('name', selected_cluster_id),
                        'cluster_description': cluster.get('description', 'cluster'),
                        'input_string': input_string,
                        'cluster_intervention_value': cluster_intervention_value,
                        'cluster_feature_count': len(cluster_features),
                        'manual_feature_count': len(manual_features_list),
                        'status': 'submitted'
                    }
                    
                    return reset_intervention_mode, hidden_dialog_style, job_info, show_progress_style, progress_content
                    
                else:
                    print(f"Condor submit failed: {result.stderr}")
                    error_content = [
                        html.Div([
                            html.Strong("❌ Job Submission Failed", style={
                                'color': '#dc2626',
                                'fontSize': '14px',
                                'marginBottom': '8px',
                                'display': 'block'
                            }),
                            html.Div([
                                html.Strong("Error: "),
                                html.Span(result.stderr or 'Unknown error occurred', style={
                                    'color': '#dc2626',
                                    'fontSize': '11px'
                                })
                            ])
                        ])
                    ]
                    
                    show_progress_style = {
                        'position': 'fixed',
                        'display': 'block',
                        'zIndex': 1003,
                        'top': '50%',
                        'left': '50%',
                        'transform': 'translate(-50%, -50%)'
                    }
                    
                    return reset_intervention_mode, hidden_dialog_style, {'success': False, 'error': result.stderr}, show_progress_style, error_content
                
            except Exception as e:
                print(f"Error submitting job: {e}")
                error_content = [
                    html.Div([
                        html.Strong("❌ Submission Error", style={
                            'color': '#dc2626',
                            'fontSize': '14px',
                            'marginBottom': '8px',
                            'display': 'block'
                        }),
                        html.Div([
                            html.Strong("Error: "),
                            html.Span(str(e), style={
                                'color': '#dc2626',
                                'fontSize': '11px'
                            })
                        ])
                    ])
                ]
                
                show_progress_style = {
                    'position': 'fixed',
                    'display': 'block',
                    'zIndex': 1003,
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)'
                }
                
                return reset_intervention_mode, hidden_dialog_style, {'success': False, 'error': str(e)}, show_progress_style, error_content
        
        return no_update, no_update, no_update, no_update, no_update

    # Handle displaying intervention results
    @app.callback(
        Output('intervention-results', 'style'),
        Output('intervention-results', 'children'),
        Input('intervention-results-store', 'data'),
        prevent_initial_call=True
    )
    def display_intervention_results(intervention_data):
        """Display intervention results in a floating panel."""
        # Always return hidden style - we don't want the top-right panel
        return {'display': 'none'}, []

    # Monitor job progress
    @app.callback(
        [Output('intervention-progress-content', 'children', allow_duplicate=True),
         Output('intervention-progress-dialog', 'style', allow_duplicate=True),
         Output('intervention-results-store', 'data', allow_duplicate=True)],
        Input('job-monitor-interval', 'n_intervals'),
        State('intervention-results-store', 'data'),
        prevent_initial_call=True
    )
    def monitor_job_progress(n_intervals, job_data):
        """Monitor Condor job progress and update the progress dialog."""
        import subprocess
        import json
        import os
        
        if not job_data or not job_data.get('job_id') or job_data.get('status') == 'completed':
            return no_update, no_update, no_update
        
        job_id = job_data['job_id']
        print(f"DEBUG: Monitoring job with ID: {job_id} (type: {type(job_id)})")
        
        try:
            # Check job status with condor_q - retry if first call fails or returns empty
            job_found = False
            job_status = None
            
            for attempt in range(3):  # Try up to 3 times
                result = subprocess.run([
                    "condor_q", "fdraye", "-format", "%d ", "ClusterId", "-format", "%s\n", "JobStatus"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    print(f"DEBUG: condor_q output (attempt {attempt + 1}): {result.stdout}")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                # Match both exact job_id and job_id as part of cluster.job format
                                cluster_id = parts[0]
                                if cluster_id == str(job_id) or cluster_id == f"{job_id}":
                                    job_found = True
                                    job_status = int(parts[1])
                                    print(f"DEBUG: Found job {job_id} with status {job_status}")
                                    break
                    if job_found:
                        break  # Job found, no need to retry
                else:
                    print(f"DEBUG: condor_q attempt {attempt + 1} failed or returned empty (returncode: {result.returncode})")
                    if attempt < 2:  # Don't sleep on last attempt
                        import time
                        time.sleep(1)  # Wait 1 second before retry
            
            if not job_found:
                # Job not found in queue, check if results are ready
                results_file = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/cluster_intervention_results.json"
                print(f"DEBUG: Job {job_id} not found in queue, checking for results file: {results_file}")
                print(f"DEBUG: File exists: {os.path.exists(results_file)}")
                if os.path.exists(results_file):
                    # Job completed successfully
                    try:
                        with open(results_file, 'r') as f:
                            intervention_results = json.load(f)
                        
                        # Update job data to completed
                        final_job_data = dict(job_data)
                        final_job_data.update({
                            'success': True,
                            'status': 'completed',
                            'results': intervention_results
                        })
                        
                        # Success content
                        success_content = [
                            html.Div([
                                html.Strong("✅ Intervention Complete!", style={
                                    'color': '#16a34a',
                                    'fontSize': '14px',
                                    'marginBottom': '10px',
                                    'display': 'block'
                                }),
                                html.Div([
                                    html.Div(f"Cluster: {job_data.get('cluster_description', 'cluster')}", style={
                                        'fontSize': '12px',
                                        'color': '#6b7280',
                                        'marginBottom': '4px'
                                    }),
                                    html.Div(f"Cluster Features: {job_data.get('cluster_feature_count', 0)}", style={
                                        'fontSize': '12px',
                                        'color': '#6b7280',
                                        'marginBottom': '4px'
                                    }),
                                    html.Div(f"Manual Features: {job_data.get('manual_feature_count', 0)}", style={
                                        'fontSize': '12px',
                                        'color': '#6b7280',
                                        'marginBottom': '4px'
                                    }),
                                    html.Div(f"Cluster Strength: ×{job_data.get('cluster_intervention_value', 0)}", style={
                                        'fontSize': '12px',
                                        'color': '#6b7280',
                                        'marginBottom': '8px'
                                    })
                                ]),
                                html.Div([
                                    html.Div(f'Input: "{job_data.get("input_string", "")}"', style={
                                        'fontSize': '11px',
                                        'fontWeight': '500',
                                        'marginBottom': '6px',
                                        'padding': '4px 8px',
                                        'backgroundColor': '#f3f4f6',
                                        'borderRadius': '4px'
                                    }),
                                    html.Div("Top predictions:", style={
                                        'fontSize': '11px',
                                        'fontWeight': '500',
                                        'marginBottom': '4px'
                                    }),
                                    html.Div([
                                        html.Div([
                                            html.Div([
                                                html.Span(f'"{token}"', style={
                                                    'fontWeight': '600',
                                                    'marginRight': '6px',
                                                    'fontSize': '11px'
                                                }),
                                                html.Span(f"{prob:.3f}", style={
                                                    'fontSize': '10px',
                                                    'color': '#1f2937',
                                                    'fontFamily': 'monospace',
                                                    'marginRight': '4px'
                                                }),
                                                html.Span(f"({'+'if diff >= 0 else ''}{diff:.3f})", style={
                                                    'fontSize': '9px',
                                                    'fontFamily': 'monospace',
                                                    'fontWeight': '600',
                                                    'color': '#16a34a' if diff > 0.01 else '#dc2626' if diff < -0.01 else '#6b7280',
                                                    'backgroundColor': 'rgba(34, 197, 94, 0.1)' if diff > 0.01 else 'rgba(239, 68, 68, 0.1)' if diff < -0.01 else 'transparent',
                                                    'padding': '1px 3px',
                                                    'borderRadius': '2px'
                                                })
                                            ], style={
                                                'display': 'flex',
                                                'alignItems': 'center',
                                                'justifyContent': 'space-between'
                                            })
                                        ], style={
                                            'marginBottom': '3px',
                                            'padding': '4px 8px',
                                            'backgroundColor': '#fef3c7' if i == 0 else '#f9fafb',
                                            'borderRadius': '4px',
                                            'fontSize': '10px',
                                            'border': '1px solid' + ('#fed7aa' if i == 0 else '#e5e7eb')
                                        }) for i, (token, prob, baseline, diff) in enumerate(zip(
                                            intervention_results.get('tokens', [])[:4], 
                                            intervention_results.get('probabilities', [])[:4],
                                            intervention_results.get('baseline_probabilities', [])[:4],
                                            intervention_results.get('probability_differences', [])[:4]
                                        ))
                                    ])
                                ]),
                                html.Div([
                                    html.Button(
                                        "Close",
                                        id="close-progress-btn",
                                        style={
                                            'backgroundColor': '#16a34a',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '6px 12px',
                                            'borderRadius': '4px',
                                            'cursor': 'pointer',
                                            'fontSize': '11px',
                                            'marginTop': '10px'
                                        }
                                    )
                                ], style={'textAlign': 'center'})
                            ])
                        ]
                        
                        return success_content, no_update, final_job_data
                        
                    except Exception as e:
                        print(f"Error reading results file: {e}")
                        # Job completed but couldn't read results
                        error_content = [
                            html.Div([
                                html.Strong("⚠️ Job Completed with Issues", style={
                                    'color': '#f59e0b',
                                    'fontSize': '14px',
                                    'marginBottom': '8px',
                                    'display': 'block'
                                }),
                                html.Div(f"Could not read results file: {str(e)}", style={
                                    'fontSize': '11px',
                                    'color': '#6b7280'
                                })
                            ])
                        ]
                        return error_content, no_update, {'success': False, 'error': 'Could not read results'}
                else:
                    # Job finished but no results file - likely failed, check error logs
                    error_log_file = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/condor_intervention.err"
                    error_details = "Job completed but no results file found"
                    
                    if os.path.exists(error_log_file):
                        try:
                            with open(error_log_file, 'r') as f:
                                error_log = f.read()
                                if error_log.strip():
                                    # Get last few lines of error log
                                    error_lines = error_log.strip().split('\n')[-10:]
                                    error_details = "Job completed but no results file found. Last error log lines:\n" + '\n'.join(error_lines)
                        except Exception as e:
                            error_details += f" (Could not read error log: {e})"
                    
                    print(f"DEBUG: Job failed - {error_details}")
                    
                    error_content = [
                        html.Div([
                            html.Strong("❌ Job Failed", style={
                                'color': '#dc2626',
                                'fontSize': '14px',
                                'marginBottom': '8px',
                                'display': 'block'
                            }),
                            html.Div(error_details, style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'whiteSpace': 'pre-wrap'
                            })
                        ])
                    ]
                    return error_content, no_update, {'success': False, 'error': 'No results file found'}
            else:
                # Job is still in queue, show status
                status_text = {
                    1: "Idle (waiting to run)",
                    2: "Running",
                    3: "Removing",
                    4: "Completed",
                    5: "Held",
                    6: "Transferring output"
                }.get(job_status, f"Unknown status ({job_status})")
                
                status_content = [
                    html.Div([
                        html.Strong("🔄 Job Running", style={
                            'color': '#0284c7',
                            'fontSize': '14px',
                            'marginBottom': '10px',
                            'display': 'block'
                        }),
                        html.Div(f"Job ID: {job_id}", style={
                            'fontSize': '11px',
                            'color': '#6b7280',
                            'fontFamily': 'monospace',
                            'marginBottom': '4px'
                        }),
                        html.Div(f"Status: {status_text}", style={
                            'fontSize': '12px',
                            'color': '#0284c7',
                            'marginBottom': '8px'
                        }),
                        html.Div([
                            html.Div(f"Cluster: {job_data.get('cluster_description', 'cluster')}", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginBottom': '2px'
                            }),
                            html.Div(f"Cluster Features: {job_data.get('cluster_feature_count', 0)}", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginBottom': '2px'
                            }),
                            html.Div(f"Manual Features: {job_data.get('manual_feature_count', 0)}", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginBottom': '2px'
                            })
                        ]),
                        html.Div("⏳ Monitoring job progress...", style={
                            'fontSize': '11px',
                            'color': '#059669',
                            'fontStyle': 'italic',
                            'marginTop': '8px'
                        })
                    ])
                ]
                
                return status_content, no_update, no_update
                
        except Exception as e:
            print(f"Error monitoring job: {e}")
            error_content = [
                html.Div([
                    html.Strong("⚠️ Monitoring Error", style={
                        'color': '#f59e0b',
                        'fontSize': '14px',
                        'marginBottom': '8px',
                        'display': 'block'
                    }),
                    html.Div(f"Error: {str(e)}", style={
                        'fontSize': '11px',
                        'color': '#6b7280'
                    })
                ])
            ]
            return error_content, no_update, no_update

    # Handle closing the progress dialog
    @app.callback(
        Output('intervention-progress-dialog', 'style', allow_duplicate=True),
        Input('close-progress-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_progress_dialog(n_clicks):
        """Close the progress dialog when Close button is clicked."""
        if n_clicks:
            return {
                'position': 'fixed',
                'display': 'none',
                'zIndex': 1003,
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)'
            }
        return no_update

    # Handle manual feature management
    @app.callback(
        [Output('manual-features-list', 'children'),
         Output('manual-features-store', 'data', allow_duplicate=True)],
        [Input('add-manual-feature-btn', 'n_clicks'),
         Input({'type': 'remove-manual-feature', 'index': ALL}, 'n_clicks')],
        State('manual-features-store', 'data'),
        prevent_initial_call=True
    )
    def manage_manual_features(add_clicks, remove_clicks, manual_features):
        """Add or remove manual features from the list."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update
            
        manual_features = manual_features or []
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        if 'add-manual-feature-btn' in trigger_id:
            # Add a new empty manual feature
            new_feature = {
                'id': len(manual_features),
                'pos': 0,
                'layer': 0, 
                'feature_idx': 0,
                'intervention_value': -10.0
            }
            manual_features.append(new_feature)
        elif 'remove-manual-feature' in trigger_id:
            # Remove the clicked feature
            import json
            trigger_data = json.loads(trigger_id.split('.')[0])
            feature_id = trigger_data['index']
            manual_features = [f for f in manual_features if f.get('id') != feature_id]
            
        # Create UI elements for each manual feature
        if not manual_features:
            feature_elements = [
                html.Div([
                    html.Span("No manual features added yet.", style={
                        'fontSize': '10px',
                        'color': '#9ca3af',
                        'fontStyle': 'italic'
                    })
                ], style={
                    'padding': '8px',
                    'textAlign': 'center',
                    'backgroundColor': '#f9fafb',
                    'borderRadius': '3px',
                    'border': '1px dashed #d1d5db'
                })
            ]
        else:
            # Add header row
            header = html.Div([
                html.Div("Pos", style={'fontSize': '8px', 'fontWeight': '600', 'color': '#6b7280', 'width': '35px', 'textAlign': 'center'}),
                html.Div("Layer", style={'fontSize': '8px', 'fontWeight': '600', 'color': '#6b7280', 'width': '35px', 'textAlign': 'center'}),
                html.Div("Feature", style={'fontSize': '8px', 'fontWeight': '600', 'color': '#6b7280', 'width': '45px', 'textAlign': 'center'}),
                html.Div("Value", style={'fontSize': '8px', 'fontWeight': '600', 'color': '#6b7280', 'width': '45px', 'textAlign': 'center'}),
                html.Div("", style={'width': '20px'})  # Space for remove button
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '4px',
                'paddingLeft': '4px'
            })
            
            feature_elements = [header]
            
            for i, feature in enumerate(manual_features):
                # Feature info display
                feature_info = html.Div([
                    html.Span(f"Feature {i+1}:", style={
                        'fontSize': '9px',
                        'fontWeight': '500',
                        'color': '#374151',
                        'marginRight': '6px'
                    }),
                    html.Span(f"pos={feature['pos']}, layer={feature['layer']}, idx={feature['feature_idx']}", style={
                        'fontSize': '8px',
                        'color': '#6b7280',
                        'fontFamily': 'monospace',
                        'backgroundColor': '#f3f4f6',
                        'padding': '1px 4px',
                        'borderRadius': '2px'
                    }),
                    html.Span(f"(×{feature['intervention_value']})", style={
                        'fontSize': '8px',
                        'color': '#dc2626' if feature['intervention_value'] < 0 else '#16a34a',
                        'fontWeight': '600',
                        'marginLeft': '4px'
                    })
                ], style={
                    'marginBottom': '2px',
                    'padding': '2px 4px'
                })
                
                # Input controls
                feature_inputs = html.Div([
                    dcc.Input(
                        id={'type': 'manual-pos', 'index': feature['id']},
                        type='number',
                        value=feature['pos'],
                        placeholder='Pos',
                        style={
                            'width': '35px',
                            'fontSize': '9px',
                            'padding': '2px 3px',
                            'marginRight': '4px',
                            'border': '1px solid #d1d5db',
                            'borderRadius': '2px',
                            'textAlign': 'center'
                        }
                    ),
                    dcc.Input(
                        id={'type': 'manual-layer', 'index': feature['id']},
                        type='number',
                        value=feature['layer'],
                        placeholder='Layer',
                        style={
                            'width': '35px',
                            'fontSize': '9px',
                            'padding': '2px 3px',
                            'marginRight': '4px',
                            'border': '1px solid #d1d5db',
                            'borderRadius': '2px',
                            'textAlign': 'center'
                        }
                    ),
                    dcc.Input(
                        id={'type': 'manual-idx', 'index': feature['id']},
                        type='number',
                        value=feature['feature_idx'],
                        placeholder='Feature',
                        style={
                            'width': '45px',
                            'fontSize': '9px',
                            'padding': '2px 3px',
                            'marginRight': '4px',
                            'border': '1px solid #d1d5db',
                            'borderRadius': '2px',
                            'textAlign': 'center'
                        }
                    ),
                    dcc.Input(
                        id={'type': 'manual-value', 'index': feature['id']},
                        type='number',
                        value=feature['intervention_value'],
                        step=0.1,
                        placeholder='Value',
                        style={
                            'width': '45px',
                            'fontSize': '9px',
                            'padding': '2px 3px',
                            'marginRight': '4px',
                            'border': '1px solid #d1d5db',
                            'borderRadius': '2px',
                            'textAlign': 'center'
                        }
                    ),
                    html.Button(
                        '×',
                        id={'type': 'remove-manual-feature', 'index': feature['id']},
                        style={
                            'backgroundColor': '#fef2f2',
                            'color': '#dc2626',
                            'border': '1px solid #fecaca',
                            'borderRadius': '2px',
                            'width': '20px',
                            'height': '20px',
                            'fontSize': '10px',
                            'cursor': 'pointer',
                            'padding': '0'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '2px'
                })
                
                # Combine info and inputs in a card-like container
                feature_element = html.Div([
                    feature_info,
                    feature_inputs
                ], style={
                    'backgroundColor': '#ffffff',
                    'border': '1px solid #e5e7eb',
                    'borderRadius': '4px',
                    'padding': '4px',
                    'marginBottom': '6px',
                    'boxShadow': '0 1px 2px rgba(0,0,0,0.05)'
                })
                
                feature_elements.append(feature_element)
            
        return feature_elements, manual_features

    # Update manual features store when inputs change
    @app.callback(
        Output('manual-features-store', 'data', allow_duplicate=True),
        [Input({'type': 'manual-pos', 'index': ALL}, 'value'),
         Input({'type': 'manual-layer', 'index': ALL}, 'value'),
         Input({'type': 'manual-idx', 'index': ALL}, 'value'),
         Input({'type': 'manual-value', 'index': ALL}, 'value')],
        State('manual-features-store', 'data'),
        prevent_initial_call=True
    )
    def update_manual_features_store(pos_values, layer_values, idx_values, intervention_values, current_features):
        """Update the manual features store when input values change."""
        if not current_features:
            return []
            
        # Update the feature values based on input changes
        updated_features = []
        for i, feature in enumerate(current_features):
            if (i < len(pos_values) and i < len(layer_values) and 
                i < len(idx_values) and i < len(intervention_values)):
                updated_feature = {
                    'id': feature['id'],
                    'pos': pos_values[i] if pos_values[i] is not None else 0,
                    'layer': layer_values[i] if layer_values[i] is not None else 0,
                    'feature_idx': idx_values[i] if idx_values[i] is not None else 0,
                    'intervention_value': intervention_values[i] if intervention_values[i] is not None else -10.0
                }
                updated_features.append(updated_feature)
                
        return updated_features


# Helper functions
def _get_default_position(cluster_id, clusters):
    """Get default position for new cluster."""
    existing_count = len(clusters)
    if existing_count <= 1:
        return {'x': 0, 'y': 0}
    
    angle = 2 * math.pi * (existing_count - 1) / max(6, existing_count)
    radius = min(100, 60 + existing_count * 15)
    
    return {
        'x': radius * math.cos(angle),
        'y': radius * math.sin(angle)
    }

def _calculate_cluster_size(nodes):
    """Calculate cluster size with maximum width and overlapping for more nodes."""
    node_count = len(nodes)
    node_width = 16
    max_width = 4 * node_width + 3 * 3 + 10  # 4 nodes with 3px gaps + padding
    cluster_height = 25
    
    if node_count <= 4:
        # Normal spacing for 4 or fewer nodes
        cluster_width = node_count * node_width + max(0, node_count - 1) * 3 + 10
    else:
        # Fixed width, nodes will overlap
        cluster_width = max_width
    
    return cluster_width, cluster_height

def _calculate_feature_positions_horizontal(nodes, cluster_pos, cluster_width, cluster_height):
    """Calculate positions for smaller feature nodes with overlapping when needed."""
    node_count = len(nodes)
    if node_count == 0:
        return []
    
    positions = []
    node_width = 12  # Match smaller node size
    padding = 3  # Smaller padding
    available_width = cluster_width - 2 * padding
    
    if node_count <= 4:
        # Normal spacing for 4 or fewer nodes
        gap = 2  # Smaller gap
        total_nodes_width = node_count * node_width + (node_count - 1) * gap
        start_x = cluster_pos['x'] - total_nodes_width / 2
        
        for i in range(node_count):
            pos_x = start_x + i * (node_width + gap)
            positions.append({'x': pos_x, 'y': cluster_pos['y']})
    else:
        # Overlapping for more than 4 nodes
        overlap_space = available_width - node_width
        spacing = overlap_space / max(1, node_count - 1)
        start_x = cluster_pos['x'] - available_width / 2
        
        for i in range(node_count):
            pos_x = start_x + i * spacing
            positions.append({'x': pos_x, 'y': cluster_pos['y']})
    
    return positions

def _calculate_connection_strength(nodes1, nodes2, graph_data):
    """Calculate connection strength between two clusters."""
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
    
    return total_weight / max(len(nodes1) * len(nodes2), 1)

def test_cluster_features(n_clusters: int = 5):
    """Test function that returns random cluster assignments."""
    # Simulate having some features to cluster
    n_features = 50  # Adjust based on your typical feature count
    
    # Generate random cluster assignments
    cluster_labels = [random.randint(0, n_clusters - 1) for _ in range(n_features)]
    
    print(f"Generated {n_features} random cluster assignments for {n_clusters} clusters")
    return cluster_labels

def create_tsne_plot(embedding_data, cluster_assignments=None):
    """Create a t-SNE/UMAP scatter plot figure with optional cluster coloring."""
    import plotly.graph_objects as go
    
    # Choose which embedding to display (prefer t-SNE, fallback to UMAP)
    embedding = None
    
    if embedding_data.get('tsne') and len(embedding_data['tsne']) > 0:
        embedding = embedding_data['tsne']
    elif embedding_data.get('umap') and len(embedding_data['umap']) > 0:
        embedding = embedding_data['umap']
    
    if not embedding:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No embedding data available",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2"
        )
    
    # Extract coordinates
    x_coords = [point[0] for point in embedding]
    y_coords = [point[1] for point in embedding]
    
    # Create scatter plot
    fig = go.Figure()
    
    # Set up colors and labels for clustering
    if cluster_assignments and 'labels' in cluster_assignments:
        cluster_labels = cluster_assignments['labels']
        
        # Define cluster colors (same as manual clustering)
        cluster_colors = [
            '#34d399',  # Strong green
            '#60a5fa',  # Strong blue  
            '#f472b6',  # Strong pink
            '#a78bfa',  # Strong purple
            '#f87171',  # Strong red
            '#fb923c',  # Strong orange
            '#22d3ee',  # Strong cyan
            '#38bdf8',  # Strong sky
            '#f472b6'   # Strong rose
        ]
        
        
        # Group points by cluster for separate traces
        cluster_points = {}
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if i < len(cluster_labels):
                cluster_id = cluster_labels[i]
                if cluster_id not in cluster_points:
                    cluster_points[cluster_id] = {'x': [], 'y': [], 'indices': []}
                cluster_points[cluster_id]['x'].append(x)
                cluster_points[cluster_id]['y'].append(y)
                cluster_points[cluster_id]['indices'].append(i)
        
        # Add one trace per cluster for better visual separation
        for cluster_id, points in cluster_points.items():
            if cluster_id >= 0:  # Skip unclustered points (cluster_id = -1)
                color = cluster_colors[cluster_id % len(cluster_colors)]
                fig.add_trace(go.Scatter(
                    x=points['x'],
                    y=points['y'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        line=dict(width=1, color=color),
                        opacity=0.8
                    ),
                    hoverinfo='skip',
                    name=f'Cluster {cluster_id + 1}',
                    customdata=points['indices']  # Store original indices for click handling
                ))
        
        # Add unclustered points if any (cluster_id = -1)
        if -1 in cluster_points:
            fig.add_trace(go.Scatter(
                x=cluster_points[-1]['x'],
                y=cluster_points[-1]['y'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(156, 163, 175, 0.7)',  # Gray for unclustered
                    line=dict(width=1, color='rgba(156, 163, 175, 1.0)'),
                    opacity=0.8
                ),
                hoverinfo='skip',
                name='Unclustered',
                customdata=cluster_points[-1]['indices']
            ))
        
    else:
        # No clustering - use single color
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=8,
                color='rgba(59, 130, 246, 0.7)',
                line=dict(width=1, color='rgba(59, 130, 246, 1.0)'),
                opacity=0.8
            ),
            hoverinfo='skip',  # Remove hover info
            name='Features',
            customdata=list(range(len(x_coords)))  # Store indices for click handling
        ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode=False,  # Disable hover mode
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            visible=False,  # Hide x-axis
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,  # Hide y-axis
            showgrid=False,
            zeroline=False
        ),
        dragmode='select'  # Enable clicking instead of panning
    )
    
    return fig
