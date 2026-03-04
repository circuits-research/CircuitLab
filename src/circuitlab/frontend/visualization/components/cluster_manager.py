from dash import html, dcc
import dash_cytoscape as cyto

class ClusterManager:
    """Enhanced cluster interface with compact horizontal node layout and edge creation."""
    
    def __init__(self):
        self.cluster_colors = [
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
    
    def create_cluster_interface(self):
        return html.Div([
            
            # Tab interface
            html.Div([
                # Tab buttons
                html.Div([
                    html.Button(
                        "Manual",
                        id="manual-tab-btn",
                        style={
                            'backgroundColor': '#f0f9ff',
                            'color': '#0284c7',
                            'border': '1px solid #bae6fd',
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
                    ),
                    html.Button(
                        "t-SNE",
                        id="tsne-tab-btn",
                        style={
                            'backgroundColor': '#f8fafc',
                            'color': '#64748b',
                            'border': '1px solid #e2e8f0',
                            'padding': '4px 10px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontSize': '10px',
                            'fontWeight': '600',
                            'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                            'minWidth': '50px',
                            'textAlign': 'center'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'backgroundColor': '#f8f9fa',
                    'paddingLeft': '16px',
                    'paddingTop': '10px',
                    'position': 'relative',
                    'zIndex': '100'
                })
            ]),
            
            # Tab content area
            html.Div([
                # Manual clustering tab content
                html.Div([
                    # Header with buttons
                    html.Div([
                        html.Div([
                            # Left side buttons
                            html.Div([
                                html.Button(
                                    "+ New",
                                    id="add-cluster-btn",
                                    style={
                                        'backgroundColor': '#f8fafc',
                                        'color': '#64748b',
                                        'border': '1px solid #e2e8f0',
                                        'padding': '4px 8px',
                                        'borderRadius': '6px',
                                        'cursor': 'pointer',
                                        'fontSize': '10px',
                                        'fontWeight': '500',
                                        'transition': 'all 0.15s ease',
                                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                                        'marginRight': '4px'
                                    }
                                ),
                                html.Button(
                                    "Link",
                                    id="link-mode-btn",
                                    style={
                                        'backgroundColor': '#f8fafc',
                                        'color': '#64748b',
                                        'border': '1px solid #e2e8f0',
                                        'padding': '4px 8px',
                                        'borderRadius': '6px',
                                        'cursor': 'pointer',
                                        'fontSize': '10px',
                                        'fontWeight': '500',
                                        'transition': 'all 0.15s ease',
                                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                                        'marginRight': '4px'
                                    }
                                ),
                                html.Button(
                                    "Delete",
                                    id="delete-mode-btn",
                                    style={
                                        'backgroundColor': '#fef2f2',
                                        'color': '#dc2626',
                                        'border': '1px solid #fecaca',
                                        'padding': '4px 8px',
                                        'borderRadius': '6px',
                                        'cursor': 'pointer',
                                        'fontSize': '10px',
                                        'fontWeight': '500',
                                        'transition': 'all 0.15s ease',
                                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)',
                                        'marginRight': '4px'
                                    }
                                ),
                                html.Button(
                                    "Intervene",
                                    id="intervention-mode-btn",
                                    style={
                                        'backgroundColor': '#fefce8',
                                        'color': '#ca8a04',
                                        'border': '1px solid #fed7aa',
                                        'padding': '4px 8px',
                                        'borderRadius': '6px',
                                        'cursor': 'pointer',
                                        'fontSize': '10px',
                                        'fontWeight': '500',
                                        'transition': 'all 0.15s ease',
                                        'boxShadow': '0 1px 2px rgba(0,0,0,0.05)'
                                    }
                                )
                            ], style={'display': 'flex', 'gap': '4px'}),
                            
                            # Right side Auto button
                            html.Button(
                                "Auto",
                                id="auto-cluster-btn",
                                style={
                                    'backgroundColor': '#f0f9ff',
                                    'color': '#0284c7',
                                    'border': '1px solid #bae6fd',
                                    'padding': '4px 8px',
                                    'borderRadius': '6px',
                                    'cursor': 'pointer',
                                    'fontSize': '10px',
                                    'fontWeight': '500',
                                    'transition': 'all 0.15s ease',
                                    'boxShadow': '0 1px 2px rgba(0,0,0,0.05)'
                                }
                            )
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
                    ], style={
                        'padding': '12px 16px',
                        'borderBottom': '1px solid #f0f0f0',
                        'backgroundColor': '#fafafa'
                    }),

                    # Cluster controls
                    html.Div([
                        html.Div([
                            html.Label("Number of clusters:", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginRight': '8px'
                            }),
                            dcc.Dropdown(
                                id='cluster-count-dropdown',
                                options=[{'label': str(i), 'value': i} for i in range(2, 21)],
                                value=5,
                                style={
                                    'width': '60px',
                                    'fontSize': '11px',
                                    'display': 'inline-block',
                                    'marginRight': '8px'
                                }
                            ),
                            html.Label("Use:", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginRight': '8px'
                            }),
                            dcc.Dropdown(
                                id='cluster-method-dropdown',
                                options=[
                                    {'label': 'Activation Rate', 'value': True},
                                    {'label': 'Normalized Score', 'value': False}
                                ],
                                value=True,
                                style={
                                    'width': '120px',
                                    'fontSize': '11px',
                                    'display': 'inline-block',
                                    'marginRight': '8px'
                                }
                            ),
                            html.Button(
                                "Create Clusters",
                                id="create-clusters-btn",
                                style={
                                    'backgroundColor': '#3b82f6',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '4px 8px',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer',
                                    'fontSize': '10px',
                                    'fontWeight': '400',
                                    'marginRight': '6px'
                                }
                            ),
                            html.Button(
                                "Cancel",
                                id="cancel-cluster-btn",
                                style={
                                    'backgroundColor': '#ffffff',
                                    'color': '#6b7280',
                                    'border': '1px solid #e5e7eb',
                                    'padding': '4px 8px',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer',
                                    'fontSize': '10px',
                                    'fontWeight': '400'
                                }
                            )
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'padding': '8px 12px',
                            'flexWrap': 'wrap',
                            'gap': '4px'
                        })
                    ], id='cluster-controls', style={'display': 'none', 'backgroundColor': '#f8f9fa'}),

                    # Mode indicator
                    html.Div(
                        id="mode-indicator",
                        style={
                            'padding': '6px 16px',
                            'fontSize': '10px',
                            'backgroundColor': '#f8f9fa',
                            'color': '#9ca3af',
                            'borderBottom': '1px solid #f0f0f0',
                            'display': 'none'
                        }
                    ),
                    
                    
                    # Cluster visualization
                    html.Div([
                        cyto.Cytoscape(
                        id='cluster-graph',
                        elements=[],
                        layout={'name': 'preset'},
                        style={'width': '100%', 'height': '400px'},  # Set explicit height
                        userZoomingEnabled=True,
                        userPanningEnabled=True,
                        boxSelectionEnabled=False,
                        zoom=0.6,  # Much smaller default view
                        minZoom=0.1,
                        maxZoom=2.0,
                        stylesheet=[
                        {
                            'selector': '.cluster-empty',
                            'style': {
                                'background-color': 'data(bgcolor)',
                                'width': 30,
                                'height': 20,
                                'shape': 'round-rectangle',
                                'label': 'data(label)',
                                'color': '#ffffff',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': 12,
                                'font-weight': 'bold',
                                'border-width': 1,
                                'border-color': '#ffffff',
                                'border-style': 'solid',
                                'opacity': 0.9
                            }
                        },
                        {
                            'selector': '.cluster-filled',
                            'style': {
                                'background-color': '#f3f4f6',
                                'width': 'data(width)',
                                'height': 'data(height)',
                                'shape': 'round-rectangle',
                                'label': '',
                                'border-width': 1,
                                'border-color': '#9ca3af',
                                'border-style': 'solid',
                                'opacity': 0.8
                            }
                        },
                        {
                            'selector': '.cluster-label',
                            'style': {
                                'background-color': 'rgba(255, 255, 255, 0.9)',
                                'width': 50,
                                'height': 18,
                                'shape': 'round-rectangle',
                                'label': 'data(description)',
                                'color': '#374151',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': 8,
                                'font-weight': '400',
                                'border-width': 1,
                                'border-color': '#e5e7eb',
                                'border-style': 'solid',
                                'opacity': 1,
                                'cursor': 'pointer',
                                'overlay-opacity': 0.1,
                                'overlay-color': '#3b82f6',
                                'events': 'yes'
                            }
                        },
                        {
                            'selector': '.cluster-highlight',
                            'style': {
                                'border-color': '#000000',
                                'border-width': 3
                            }
                        },
                        {
                            'selector': '.cluster-delete',
                            'style': {
                                'border-color': '#dc2626',
                                'border-width': 3,
                                'opacity': 0.7
                            }
                        },
                        {
                            'selector': '.feature-node',
                            'style': {
                                'background-color': 'data(bgcolor)',
                                'width': 12,
                                'height': 12,
                                'shape': 'ellipse',
                                'border-width': 1,
                                'border-color': '#ffffff',
                                'border-style': 'solid',
                                'label': '',
                                'opacity': 0.9
                            }
                        },
                        {
                            'selector': '.cluster-edge-auto',
                            'style': {
                                'line-color': '#9ca3af',
                                'width': 1,
                                'opacity': 0.5,
                                'curve-style': 'bezier'
                            }
                        },
                        {
                            'selector': '.cluster-edge-manual',
                            'style': {
                                'line-color': '#3b82f6',
                                'width': 2,
                                'opacity': 0.8,
                                'curve-style': 'bezier'
                            }
                        },
                        {
                            'selector': '.cluster-input',
                            'style': {
                                'background-color': '#ffffff',
                                'width': 'data(width)',
                                'height': 25,
                                'shape': 'rectangle',
                                'label': 'data(text)',
                                'color': '#374151',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': 9,
                                'font-weight': 'normal',
                                'border-width': 1,
                                'border-color': '#d1d5db',
                                'border-style': 'solid',
                                'opacity': 1.0,
                                'text-wrap': 'wrap',
                                'text-max-width': 'data(width)'
                            }
                        }
                        ]
                        )
                    ], style={
                        'width': '100%',
                        'height': '400px',
                        'backgroundColor': '#fafafa',
                        'border': '1px solid #e5e7eb',
                        'position': 'relative'
                    })
                ], id='manual-tab-content', style={
                    'display': 'block',
                    'height': '100%',
                    'backgroundColor': 'white'
                }),
                
                # t-SNE tab content
                html.Div([
                    # t-SNE clustering controls
                    html.Div([
                        html.Div([
                            html.Label("Apply clustering:", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginRight': '8px'
                            }),
                            dcc.Dropdown(
                                id='tsne-cluster-count-dropdown',
                                options=[{'label': 'No clustering', 'value': 0}] + 
                                        [{'label': f'{i} clusters', 'value': i} for i in range(2, 21)],
                                value=0,
                                style={
                                    'width': '120px',
                                    'fontSize': '11px',
                                    'display': 'inline-block',
                                    'marginRight': '8px'
                                }
                            ),
                            html.Label("Method:", style={
                                'fontSize': '11px',
                                'color': '#6b7280',
                                'marginRight': '8px'
                            }),
                            dcc.Dropdown(
                                id='tsne-cluster-method-dropdown',
                                options=[
                                    {'label': 'Activation Rate', 'value': True},
                                    {'label': 'Normalized Score', 'value': False}
                                ],
                                value=True,
                                style={
                                    'width': '120px',
                                    'fontSize': '11px',
                                    'display': 'inline-block'
                                }
                            )
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'padding': '8px 12px',
                            'flexWrap': 'wrap',
                            'gap': '4px'
                        })
                    ], style={
                        'borderBottom': '1px solid #f0f0f0',
                        'backgroundColor': '#fafafa'
                    }),
                    
                    # t-SNE plot
                    dcc.Graph(
                        id='tsne-plot',
                        style={'width': '100%', 'height': 'calc(100% - 60px)'},
                        config={
                            'displayModeBar': False,
                            'doubleClick': 'reset',
                            'scrollZoom': True,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                        }
                    )
                ], id='tsne-tab-content', style={
                    'display': 'none',
                    'height': '100%',
                    'backgroundColor': 'white'
                })
            ], style={
                'flex': '1',
                'minHeight': '0',
                'backgroundColor': '#fafafa',
                'position': 'relative',
                'border': '1px solid #e5e7eb',
                'borderTop': 'none'
            }),

            # Stores for state
            dcc.Store(id='clusters-store', data={}),
            dcc.Store(id='cluster-positions-store', data={}),
            dcc.Store(id='manual-edges-store', data=[]),
            dcc.Store(id='link-mode-store', data={'active': False, 'first_cluster': None}),
            dcc.Store(id='delete-mode-store', data={'active': False}),
            dcc.Store(id='tsne-toggle-store', data={'active': False}),
            dcc.Store(id='embedding-data-store', data={}),
            dcc.Store(id='active-tab-store', data={'tab': 'manual'}),
            dcc.Store(id='tsne-clusters-store', data={}),
            
            # Floating text input for editing cluster labels
            html.Div([
                dcc.Input(
                    id='cluster-label-input',
                    type='text',
                    placeholder='Type cluster name',
                    style={
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
                )
            ]),
            
            # Enhanced intervention dialog with manual feature selection
            html.Div([
                html.Div([
                    html.Div("Configure Intervention", style={
                        'fontSize': '14px',
                        'fontWeight': '600',
                        'marginBottom': '12px',
                        'color': '#374151'
                    }),
                    
                    # Cluster features section
                    html.Div([
                        html.Div("Cluster Features", style={
                            'fontSize': '12px',
                            'fontWeight': '500',
                            'marginBottom': '4px',
                            'color': '#374151'
                        }),
                        html.Div("Apply to all features in selected cluster:", style={
                            'fontSize': '10px',
                            'color': '#6b7280',
                            'marginBottom': '6px'
                        }),
                        dcc.Input(
                            id='cluster-intervention-strength',
                            type='number',
                            value=-10.0,
                            step=0.1,
                            placeholder='Value',
                            style={
                                'width': '80px',
                                'fontSize': '11px',
                                'padding': '4px 6px',
                                'border': '1px solid #d1d5db',
                                'borderRadius': '4px',
                                'marginBottom': '8px'
                            }
                        )
                    ], style={
                        'marginBottom': '12px',
                        'padding': '8px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '4px',
                        'border': '1px solid #e5e7eb'
                    }),
                    
                    # Manual features section
                    html.Div([
                        html.Div("Manual Features (Optional)", style={
                            'fontSize': '12px',
                            'fontWeight': '500',
                            'marginBottom': '4px',
                            'color': '#374151'
                        }),
                        html.Div("Add individual features:", style={
                            'fontSize': '10px',
                            'color': '#6b7280',
                            'marginBottom': '6px'
                        }),
                        html.Div(id='manual-features-list', children=[], style={
                            'marginBottom': '8px'
                        }),
                        html.Button(
                            "+ Add Feature",
                            id="add-manual-feature-btn",
                            style={
                                'backgroundColor': '#f3f4f6',
                                'color': '#4b5563',
                                'border': '1px solid #d1d5db',
                                'padding': '3px 6px',
                                'borderRadius': '3px',
                                'cursor': 'pointer',
                                'fontSize': '9px',
                                'fontWeight': '400',
                                'marginBottom': '8px'
                            }
                        )
                    ], style={
                        'marginBottom': '12px',
                        'padding': '8px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '4px',
                        'border': '1px solid #e5e7eb'
                    }),
                    
                    # Freeze attention section
                    html.Div([
                        html.Div("Advanced Options", style={
                            'fontSize': '12px',
                            'fontWeight': '500',
                            'marginBottom': '6px',
                            'color': '#374151'
                        }),
                        html.Div([
                            dcc.Checklist(
                                id='freeze-attention-toggle',
                                options=[{'label': ' Freeze attention patterns during intervention', 'value': 'freeze'}],
                                value=['freeze'],  # Default to checked (True)
                                style={
                                    'fontSize': '10px',
                                    'color': '#4b5563'
                                },
                                persistence=True,
                                persistence_type='session'
                            ),
                        ])
                    ], style={
                        'marginBottom': '12px',
                        'padding': '8px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '4px',
                        'border': '1px solid #e5e7eb'
                    }),
                    
                    # Action buttons
                    html.Div([
                        html.Button(
                            "Run Intervention",
                            id="run-intervention-btn",
                            style={
                                'backgroundColor': '#ca8a04',
                                'color': 'white',
                                'border': 'none',
                                'padding': '6px 12px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'fontSize': '11px',
                                'fontWeight': '500',
                                'marginRight': '6px'
                            }
                        ),
                        html.Button(
                            "Cancel",
                            id="cancel-intervention-btn",
                            style={
                                'backgroundColor': '#ffffff',
                                'color': '#6b7280',
                                'border': '1px solid #d1d5db',
                                'padding': '6px 12px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'fontSize': '11px',
                                'fontWeight': '400'
                            }
                        )
                    ], style={'textAlign': 'center'})
                ], style={
                    'padding': '16px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'border': '1px solid #d1d5db',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                    'minWidth': '320px'
                })
            ], id='intervention-dialog', style={
                'position': 'fixed',
                'display': 'none',
                'zIndex': 1001,
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)'
            }),
            
            # Intervention results display
            html.Div(id='intervention-results', style={
                'position': 'fixed',
                'display': 'none',
                'zIndex': 1002,
                'top': '20px',
                'right': '20px',
                'maxWidth': '300px',
                'padding': '12px',
                'backgroundColor': 'white',
                'borderRadius': '6px',
                'border': '1px solid #d1d5db',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
                'fontSize': '11px'
            }),
            
            # Progress dialog for job monitoring
            html.Div([
                html.Div(
                    id='intervention-progress-content',
                    children=[],
                    style={
                        'padding': '16px',
                        'backgroundColor': 'white',
                        'borderRadius': '8px',
                        'border': '1px solid #d1d5db',
                        'boxShadow': '0 6px 20px rgba(0,0,0,0.15)',
                        'minWidth': '300px',
                        'maxWidth': '400px'
                    }
                )
            ], id='intervention-progress-dialog', style={
                'position': 'fixed',
                'display': 'none',
                'zIndex': 1003,
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)'
            }),
            
            # Interval component for job monitoring (checks every 1 second)
            dcc.Interval(
                id='job-monitor-interval',
                interval=1000,  # 1 second
                n_intervals=0,
                disabled=False
            ),
            
            dcc.Store(id='editing-cluster-id', data=None),
            dcc.Store(id='intervention-mode-store', data={'active': False, 'selected_cluster': None}),
            dcc.Store(id='intervention-results-store', data=None),
            dcc.Store(id='manual-features-store', data=[])
            
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'height': '100%',
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'border': '1px solid #e5e7eb',
            'overflow': 'hidden'
        })
