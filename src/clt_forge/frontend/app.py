import dash
from dash import html, dcc
import traceback

from .config.settings import AppConfig
from .data.loaders import DataLoader
from .visualization.graph.layout import GraphLayoutCalculator
from .visualization.components.graph_component import GraphComponent
from .visualization.components.cluster_manager import ClusterManager
from .callbacks.graph_callbacks import register_callbacks
from .callbacks.cluster_callbacks import register_cluster_callbacks
from .callbacks.annotation_callbacks import register_annotation_callbacks
from .callbacks.state_callbacks import register_state_callbacks

def create_app(config: AppConfig) -> dash.Dash:
    """Create and configure the Dash application."""
    try:        
        # Initialize components
        data_loader = DataLoader(config)
        layout_calculator = GraphLayoutCalculator(config.graph, data_loader)
        graph_component = GraphComponent(config, data_loader, layout_calculator)
        
        cluster_manager = ClusterManager()
        
        # Pre-load graph data (but don't precompute figures until browser loads)
        _ = graph_component.get_graph_data()
        
        # Create optimized app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    /* Performance optimizations */
                    .dash-loading { display: none !important; }
                    * { box-sizing: border-box; }
                    
                    /* Neuronpedia-style cluster interface */
                    #add-cluster-btn:hover {
                        background-color: #e5e7eb !important;
                        border-color: #9ca3af !important;
                    }
                    
                    div[id*="cluster-item"]:hover {
                        background-color: #f0f0f0 !important;
                        border-color: #d1d5db !important;
                    }
                    
                    #drop-zone:hover {
                        background-color: #f3f4f6 !important;
                        border-color: #9ca3af !important;
                    }
                    
                    /* Delete button hover */
                    button[id*="delete-cluster"]:hover {
                        color: #ef4444 !important;
                    }
                    
                    /* Smooth transitions */
                    div[id*="cluster-item"], #drop-zone, #add-cluster-btn {
                        transition: all 0.2s ease;
                    }
                    
                    /* Draggable annotation styles */
                    .draggable-annotation {
                        transition: box-shadow 0.2s ease;
                    }
                    
                    .draggable-annotation:hover {
                        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
                    }
                    
                    .draggable-annotation.dragging {
                        box-shadow: 0 8px 16px rgba(0,0,0,0.25) !important;
                        z-index: 1002 !important;
                    }
                    
                    .drag-handle:hover {
                        color: #374151 !important;
                        background-color: rgba(59, 130, 246, 0.1) !important;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

        app.layout = create_layout(graph_component, cluster_manager, config)
        
        # Register callbacks
        print("Registering callbacks...")
        register_callbacks(app, graph_component)
        register_cluster_callbacks(app, graph_component, cluster_manager)
        register_annotation_callbacks(app, graph_component)
        register_state_callbacks(app, graph_component)
        
        print("App created successfully!")
        return app
        
    except Exception as e:
        print(f"Error creating app: {e}")
        traceback.print_exc()
        raise

def create_layout(graph_component: GraphComponent, cluster_manager: ClusterManager, 
                 config: AppConfig) -> html.Div:
    """Create the main app layout."""
    try:
        print("Creating layout...")
        
        # Test graph component creation
        graph_container = graph_component.create_graph_container()
        print("Graph container created successfully")
        
        # Get some stats for the header
        graph_data = graph_component.get_graph_data()

        return html.Div([
            # Header - same as before
            html.Div([
                html.Div([
                    # Left side - Logo
                    html.Div([
                        html.Img(
                            src="/assets/max_planck_logo.jpg",
                            style={
                                'height': '32px',
                                'width': 'auto'
                            }
                        )
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'flex': '1'
                    }),
                    
                    # Right side - Model and stats info
                    html.Div([
                        html.Div([
                            html.Span("GPT-2 Small", style={
                                'fontSize': '13px',
                                'backgroundColor': '#3b82f6',
                                'color': 'white',
                                'padding': '2px 8px',
                                'borderRadius': '10px',
                                'fontWeight': '500',
                                'marginBottom': '3px',
                                'display': 'inline-block'
                            })
                        ], style={'marginBottom': '3px'}),
                        html.Div([
                            html.Span(f"{len(graph_data.nodes)}", style={
                                'fontSize': '13px',
                                'fontWeight': '600',
                                'color': '#059669',
                                'marginRight': '4px'
                            }),
                            html.Span("features", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginRight': '12px'
                            }),
                            html.Span(f"{len(graph_data.edges)}", style={
                                'fontSize': '13px',
                                'fontWeight': '600',
                                'color': '#3b82f6',
                                'marginRight': '4px'
                            }),
                            html.Span("connections", style={
                                'fontSize': '11px',
                                'color': '#6b7280'
                            })
                        ])
                    ], style={
                        'textAlign': 'right',
                        'lineHeight': '1.1'
                    })
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'space-between',
                    'padding': '16px 24px',
                    'backgroundColor': 'white',
                    'borderBottom': '1px solid #e2e8f0'
                })
            ], style={
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'right': '0',
                'zIndex': '1000',
                'height': '72px'
            }),
            
            # Main content area
            html.Div([
                # Graph container (left side)
                html.Div([
                    graph_container
                ], style={
                    'position': 'absolute',
                    'left': '0',
                    'top': '0',
                    'width': '68%',
                    'height': '100%',
                    'backgroundColor': config.graph.background_color
                }),
                
                # Feature display (top right - reduced to 30%)
                html.Div([
                    html.Div(id='activation-display')
                ], style={
                    'position': 'absolute',
                    'right': '0',
                    'top': '0',
                    'width': '32%',
                    'height': '65%',  # Increased from 50% to 65% for feature analysis
                    'backgroundColor': '#f8fafc',
                    'borderLeft': '1px solid #e2e8f0',
                    'overflowY': 'auto'
                }),
                
                # Cluster interface (bottom right - reduced to 35%)
                html.Div([
                    cluster_manager.create_cluster_interface()
                ], style={
                    'position': 'absolute',
                    'right': '0',
                    'bottom': '0',
                    'width': '32%',
                    'height': '43.7%',  # Reduced from 50% to 35%
                    'backgroundColor': '#f8fafc',
                    'borderLeft': '1px solid #e2e8f0',
                    'borderTop': '1px solid #e2e8f0'
                })
            ], style={
                'position': 'fixed',
                'top': '72px',
                'left': '0',
                'right': '0',
                'bottom': '0',
                'overflow': 'hidden'
            }),
            
            # Stores for state management
            dcc.Store(id='selected-feature-store'),
            dcc.Store(id='clicked-nodes-store'),
            dcc.Store(id='click-detection-store'),
            dcc.Store(id='selected-cluster-store', data=[]),  # Initialize as empty list instead of None
            dcc.Store(id='annotation-mode-store', data={'active': False}),
            dcc.Store(id='text-annotations-store', data={}),
            dcc.Store(id='viewport-size-store'),  # Store for screen dimensions
            
            # Hidden div to trigger viewport detection
            html.Div(id='viewport-trigger', style={'display': 'none'})
            
        ], style={
            'backgroundColor': 'white',
            'height': '100vh',
            'fontFamily': 'Inter, Arial, sans-serif',
            'overflow': 'hidden'
        })
    
    except Exception as e:
        print(f"Error creating layout: {e}")
        traceback.print_exc()
        # Return a simple error layout
        return html.Div([
            html.H1("Error Loading Application"),
            html.P(f"Error: {str(e)}"),
            html.Pre(traceback.format_exc())
        ], style={'padding': '20px'})


def main(config: AppConfig):
    try:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

        app = create_app(config)

        debug_mode = config.debug or True

        print(f"Starting server on {config.host}:{config.port} (debug={debug_mode})")
        app.run(
            debug=debug_mode,
            host=config.host,
            port=config.port,
            use_reloader=True
        )
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
