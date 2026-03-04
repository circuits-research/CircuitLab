import json
import base64
from dash import Input, Output, State, no_update

def register_state_callbacks(app, graph_component):
    """Register all state save/load related callbacks."""
    
    @app.callback(
        [Output('clusters-store', 'data', allow_duplicate=True),
         Output('text-annotations-store', 'data', allow_duplicate=True),
         Output('selected-feature-store', 'data', allow_duplicate=True),
         Output('selected-cluster-store', 'data', allow_duplicate=True),
         Output('annotation-mode-store', 'data', allow_duplicate=True),
         Output('cluster-positions-store', 'data', allow_duplicate=True),
         Output('manual-edges-store', 'data', allow_duplicate=True),
         Output('link-mode-store', 'data', allow_duplicate=True),
         Output('delete-mode-store', 'data', allow_duplicate=True),
         Output('tsne-toggle-store', 'data', allow_duplicate=True),
         Output('embedding-data-store', 'data', allow_duplicate=True),
         Output('active-tab-store', 'data', allow_duplicate=True),
         Output('tsne-clusters-store', 'data', allow_duplicate=True),
         Output('intervention-mode-store', 'data', allow_duplicate=True),
         Output('intervention-results-store', 'data', allow_duplicate=True),
         Output('manual-features-store', 'data', allow_duplicate=True)],
        Input('load-state-upload', 'contents'),
        State('load-state-upload', 'filename'),
        prevent_initial_call=True
    )
    def load_dashboard_state(contents, filename):
        """Load the complete dashboard state from uploaded JSON file."""
        
        if not contents:
            return (no_update,) * 16
        
        try:
            # Decode the uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            state_data = json.loads(decoded.decode('utf-8'))
            
            # Extract all the state components
            clusters_data = state_data.get('clusters', {})
            annotations_data = state_data.get('text_annotations', {})
            
            # Force all annotations to non-editing mode when loaded
            for annotation_id, annotation in annotations_data.items():
                annotation['editing'] = False
            selected_feature = state_data.get('selected_feature', None)
            selected_clusters = state_data.get('selected_clusters', [])
            annotation_mode = state_data.get('annotation_mode', {'active': False})
            
            # Extract cluster and interface state
            cluster_positions = state_data.get('cluster_positions', {})
            manual_edges = state_data.get('manual_edges', [])
            link_mode = state_data.get('link_mode', {'active': False, 'first_cluster': None})
            delete_mode = state_data.get('delete_mode', {'active': False})
            tsne_toggle = state_data.get('tsne_toggle', {'active': False})
            embedding_data = state_data.get('embedding_data', {})
            active_tab = state_data.get('active_tab', {'tab': 'manual'})
            tsne_clusters = state_data.get('tsne_clusters', {})
            intervention_mode = state_data.get('intervention_mode', {'active': False, 'selected_cluster': None})
            intervention_results = state_data.get('intervention_results', None)
            manual_features = state_data.get('manual_features', [])
            
            print(f"Loaded state from {filename}: {len(clusters_data)} clusters, {len(annotations_data)} annotations, {len(cluster_positions)} positions, {len(manual_edges)} edges")
            
            return (clusters_data, annotations_data, selected_feature, 
                   selected_clusters, annotation_mode, cluster_positions,
                   manual_edges, link_mode, delete_mode, tsne_toggle,
                   embedding_data, active_tab, tsne_clusters, intervention_mode,
                   intervention_results, manual_features)
            
        except Exception as e:
            print(f"Error loading state: {e}")
            return (no_update,) * 16
    
    # Clientside callback to trigger file upload when load button is clicked
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks) {
                // Find the hidden upload component and trigger click
                const uploadInput = document.querySelector('#load-state-upload input[type="file"]');
                if (uploadInput) {
                    uploadInput.click();
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('load-state-btn', 'style', allow_duplicate=True),
        Input('load-state-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    
    # Use clientside callback for saving to create and download the file
    app.clientside_callback(
        """
        function(n_clicks, clusters_data, annotations_data, selected_feature, selected_clusters, annotation_mode,
                cluster_positions, manual_edges, link_mode, delete_mode, tsne_toggle, embedding_data,
                active_tab, tsne_clusters, intervention_mode, intervention_results, manual_features) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            
            // Collect all state data  
            const state_data = {
                version: '2.0',
                timestamp: new Date().toISOString(),
                clusters: clusters_data || {},
                text_annotations: annotations_data || {},
                selected_feature: selected_feature,
                selected_clusters: selected_clusters || [],
                annotation_mode: annotation_mode || {active: false},
                cluster_positions: cluster_positions || {},
                manual_edges: manual_edges || [],
                link_mode: link_mode || {active: false, first_cluster: null},
                delete_mode: delete_mode || {active: false},
                tsne_toggle: tsne_toggle || {active: false},
                embedding_data: embedding_data || {},
                active_tab: active_tab || {tab: 'manual'},
                tsne_clusters: tsne_clusters || {},
                intervention_mode: intervention_mode || {active: false, selected_cluster: null},
                intervention_results: intervention_results,
                manual_features: manual_features || []
            };
            
            // Create and download the file
            const json_string = JSON.stringify(state_data, null, 2);
            const blob = new Blob([json_string], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const filename = `featflow_state_${timestamp}.json`;
            
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log(`Saved state: ${Object.keys(state_data.clusters || {}).length} clusters, ${Object.keys(state_data.text_annotations || {}).length} annotations, ${Object.keys(state_data.cluster_positions || {}).length} positions, ${(state_data.manual_edges || []).length} edges`);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('save-state-btn', 'style', allow_duplicate=True),
        [Input('save-state-btn', 'n_clicks')],
        [State('clusters-store', 'data'),
         State('text-annotations-store', 'data'),
         State('selected-feature-store', 'data'),
         State('selected-cluster-store', 'data'),
         State('annotation-mode-store', 'data'),
         State('cluster-positions-store', 'data'),
         State('manual-edges-store', 'data'),
         State('link-mode-store', 'data'),
         State('delete-mode-store', 'data'),
         State('tsne-toggle-store', 'data'),
         State('embedding-data-store', 'data'),
         State('active-tab-store', 'data'),
         State('tsne-clusters-store', 'data'),
         State('intervention-mode-store', 'data'),
         State('intervention-results-store', 'data'),
         State('manual-features-store', 'data')],
        prevent_initial_call=True
    )
