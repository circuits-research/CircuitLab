import uuid
from dash import Input, Output, State, no_update, callback_context, html, dcc, ALL

def register_annotation_callbacks(app, graph_component):
    """Register all text annotation related callbacks."""
    
    # Force refresh annotations when state is loaded
    @app.callback(
        Output('text-annotations-container', 'children', allow_duplicate=True),
        Input('text-annotations-store', 'data'),
        prevent_initial_call='initial_duplicate'  # Allow initial call for duplicate callbacks
    )
    def refresh_annotations_on_load(annotations_data):
        """Force refresh annotations when loaded from saved state."""
        if not annotations_data:
            return []
        
        annotation_elements = []
        
        for annotation_id, annotation in annotations_data.items():
            # Use coordinates as percentages for consistent positioning
            pixel_x = f"{max(0, min(annotation['x'], 95))}%"
            pixel_y = f"{max(0, min(annotation['y'], 95))}%"
            
            # Check if annotation is in editing mode
            if annotation.get('editing', False):
                # Editable text input - same as original callback
                element = html.Div([
                    # Tiny drag handle
                    html.Div('⋮', style={
                        'cursor': 'move',
                        'padding': '0px 2px',
                        'color': '#9ca3af',
                        'fontSize': '8px',
                        'userSelect': 'none',
                        'marginRight': '2px',
                        'lineHeight': '1'
                    }, className='drag-handle'),
                    dcc.Input(
                        id={'type': 'annotation-input', 'index': annotation_id},
                        value=annotation['text'],
                        style={
                            'border': 'none',
                            'outline': '1px solid #3b82f6',
                            'borderRadius': '1px',
                            'padding': '1px 2px',
                            'fontSize': f"{annotation.get('font_size', 10)}px",
                            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                            'color': annotation.get('color', '#374151'),
                            'width': 'auto',
                            'minWidth': '40px',
                            'fontFamily': 'inherit'
                        },
                        autoFocus=True
                    ),
                    html.Button('✓', 
                        id={'type': 'annotation-save', 'index': annotation_id},
                        style={
                            'marginLeft': '2px',
                            'padding': '1px 3px',
                            'fontSize': '8px',
                            'border': 'none',
                            'backgroundColor': '#10b981',
                            'color': 'white',
                            'borderRadius': '1px',
                            'cursor': 'pointer',
                            'lineHeight': '1'
                        }
                    ),
                    html.Button('✕',
                        id={'type': 'annotation-delete', 'index': annotation_id},
                        style={
                            'marginLeft': '2px',
                            'padding': '1px 3px',
                            'fontSize': '8px',
                            'border': 'none',
                            'backgroundColor': '#ef4444',
                            'color': 'white',
                            'borderRadius': '1px',
                            'cursor': 'pointer',
                            'lineHeight': '1'
                        }
                    )
                ], 
                id={'type': 'annotation-div', 'index': annotation_id},
                style={
                    'position': 'absolute',
                    'left': pixel_x,
                    'top': pixel_y,
                    'display': 'flex',
                    'alignItems': 'center',
                    'zIndex': '1000',
                    'pointerEvents': 'auto',
                    'transform': 'translate(-50%, -50%)'
                },
                className='draggable-annotation editing'
                )
            else:
                # Display mode - same as original callback
                element = html.Div([
                    # Tiny drag handle
                    html.Div('⋮', style={
                        'cursor': 'move',
                        'padding': '0px 2px',
                        'color': '#9ca3af',
                        'fontSize': '8px',
                        'userSelect': 'none',
                        'marginRight': '2px',
                        'lineHeight': '1'
                    }, className='drag-handle'),
                    html.Div(annotation['text'], style={
                        'display': 'inline-block',
                        'padding': '1px 3px',
                        'fontSize': f"{annotation.get('font_size', 10)}px",
                        'color': annotation.get('color', '#374151'),
                        'backgroundColor': 'rgba(255, 255, 255, 0.9)',
                        'border': '1px solid rgba(0,0,0,0.1)',
                        'borderRadius': '2px',
                        'cursor': 'pointer',
                        'fontFamily': 'inherit',
                        'userSelect': 'none'
                    }),
                    html.Button('✕',
                        id={'type': 'annotation-delete', 'index': annotation_id},
                        style={
                            'marginLeft': '2px',
                            'padding': '1px 3px',
                            'fontSize': '8px',
                            'border': 'none',
                            'backgroundColor': '#ef4444',
                            'color': 'white',
                            'borderRadius': '1px',
                            'cursor': 'pointer',
                            'lineHeight': '1'
                        }
                    )
                ], 
                id={'type': 'annotation-div', 'index': annotation_id},
                style={
                    'position': 'absolute',
                    'left': pixel_x,
                    'top': pixel_y,
                    'display': 'flex',
                    'alignItems': 'center',
                    'zIndex': '1000',
                    'pointerEvents': 'auto',
                    'transform': 'translate(-50%, -50%)'
                },
                className='draggable-annotation'
                )
            
            annotation_elements.append(element)
        
        return annotation_elements
    
    @app.callback(
        [Output('annotation-mode-store', 'data'),
         Output('annotation-mode-btn', 'style'),
         Output('annotation-mode-indicator', 'children'),
         Output('annotation-mode-indicator', 'style')],
        Input('annotation-mode-btn', 'n_clicks'),
        State('annotation-mode-store', 'data'),
        prevent_initial_call=True
    )
    def toggle_annotation_mode(n_clicks, annotation_mode_data):
        """Toggle annotation mode on/off."""
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        
        current_active = annotation_mode_data.get('active', False) if annotation_mode_data else False
        new_active = not current_active
        
        # Button style
        button_style = {
            'backgroundColor': '#3b82f6' if new_active else '#ffffff',
            'color': 'white' if new_active else '#6b7280',
            'border': f'1px solid {"#3b82f6" if new_active else "#e5e7eb"}',
            'padding': '6px 12px',
            'borderRadius': '6px',
            'cursor': 'pointer',
            'fontSize': '11px',
            'fontWeight': '500',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'transition': 'all 0.2s ease',
            'display': 'flex',
            'alignItems': 'center',
            'gap': '4px'
        }
        
        # Mode indicator
        if new_active:
            indicator_text = "Annotation Mode: Click anywhere on the graph to add text"
            indicator_style = {
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
                'display': 'block'
            }
        else:
            indicator_text = ""
            indicator_style = {
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
        
        return {'active': new_active}, button_style, indicator_text, indicator_style
    
    @app.callback(
        Output('text-annotations-store', 'data', allow_duplicate=True),
        Input('annotation-mode-btn', 'n_clicks'),
        [State('annotation-mode-store', 'data'),
         State('text-annotations-store', 'data')],
        prevent_initial_call=True
    )
    def create_annotation_on_button_click(n_clicks, annotation_mode_data, annotations_data):
        """Create a new text annotation next to the button when clicked."""
        
        if not n_clicks:
            return no_update
        
        # Only create annotation if we're entering annotation mode (not exiting)
        current_active = annotation_mode_data.get('active', False) if annotation_mode_data else False
        if current_active:  # If already active, this click is to deactivate
            return no_update
        
        # Create new annotation next to the button
        annotation_id = str(uuid.uuid4())
        new_annotation = {
            'id': annotation_id,
            'x': 15,  # Position next to button (percentage)
            'y': 5,   # Top position (percentage)
            'text': 'New text',  # Default text
            'editing': True,  # Start in editing mode
            'font_size': 10,
            'color': '#374151'
        }
        
        # Add to annotations store
        annotations_data = annotations_data or {}
        annotations_data[annotation_id] = new_annotation
        
        return annotations_data
    
    @app.callback(
        Output('text-annotations-container', 'children'),
        [Input('text-annotations-store', 'data'),
         Input('active-feature-graph', 'figure')],
        prevent_initial_call=True
    )
    def update_annotation_overlay(annotations_data, figure):
        """Update the text annotations overlay based on stored annotations."""
        
        if not annotations_data:
            return []
        
        annotation_elements = []
        
        for annotation_id, annotation in annotations_data.items():
            # Use coordinates as percentages for consistent positioning
            pixel_x = f"{max(0, min(annotation['x'], 95))}%"
            pixel_y = f"{max(0, min(annotation['y'], 95))}%"
            
            if annotation.get('editing', False):
                # Editable text input - minimal design
                element = html.Div([
                    # Tiny drag handle
                    html.Div('⋮', style={
                        'cursor': 'move',
                        'padding': '0px 2px',
                        'color': '#9ca3af',
                        'fontSize': '8px',
                        'userSelect': 'none',
                        'marginRight': '2px',
                        'lineHeight': '1'
                    }, className='drag-handle'),
                    dcc.Input(
                        id={'type': 'annotation-input', 'index': annotation_id},
                        value=annotation['text'],
                        style={
                            'border': 'none',
                            'outline': '1px solid #3b82f6',
                            'borderRadius': '1px',
                            'padding': '1px 2px',
                            'fontSize': f"{annotation['font_size']}px",
                            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                            'color': annotation['color'],
                            'width': 'auto',
                            'minWidth': '40px',
                            'fontFamily': 'inherit'
                        },
                        autoFocus=True
                    ),
                    html.Button('✓', 
                        id={'type': 'annotation-save', 'index': annotation_id},
                        style={
                            'marginLeft': '2px',
                            'padding': '1px 3px',
                            'fontSize': '8px',
                            'border': 'none',
                            'backgroundColor': '#10b981',
                            'color': 'white',
                            'borderRadius': '1px',
                            'cursor': 'pointer',
                            'lineHeight': '1'
                        }
                    ),
                    html.Button('✕', 
                        id={'type': 'annotation-delete', 'index': annotation_id},
                        style={
                            'marginLeft': '1px',
                            'padding': '1px 3px',
                            'fontSize': '8px',
                            'border': 'none',
                            'backgroundColor': '#ef4444',
                            'color': 'white',
                            'borderRadius': '1px',
                            'cursor': 'pointer',
                            'lineHeight': '1'
                        }
                    )
                ], 
                id={'type': 'annotation-box', 'index': annotation_id},
                className='draggable-annotation',
                style={
                    'position': 'absolute',
                    'left': pixel_x,
                    'top': pixel_y,
                    'pointerEvents': 'auto',
                    'zIndex': '1001',
                    'display': 'flex',
                    'alignItems': 'center',
                    'padding': '1px',
                    'gap': '1px'
                })
            else:
                # Static text display - minimal
                element = html.Div([
                    # Tiny drag handle
                    html.Div('⋮', style={
                        'cursor': 'move',
                        'padding': '0px 1px',
                        'color': '#d1d5db',
                        'fontSize': '6px',
                        'userSelect': 'none',
                        'marginRight': '2px',
                        'lineHeight': '1',
                        'opacity': '0.7'
                    }, className='drag-handle'),
                    html.Span(
                        annotation['text'],
                        id={'type': 'annotation-text', 'index': annotation_id},
                        style={
                            'fontSize': f"{annotation['font_size']}px",
                            'color': annotation['color'],
                            'cursor': 'pointer',
                            'userSelect': 'none',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                            'padding': '1px 2px',
                            'borderRadius': '1px',
                            'border': '1px solid rgba(0,0,0,0.1)',
                            'lineHeight': '1.2'
                        }
                    )
                ],
                id={'type': 'annotation-box', 'index': annotation_id},
                className='draggable-annotation',
                style={
                    'position': 'absolute',
                    'left': pixel_x,
                    'top': pixel_y,
                    'pointerEvents': 'auto',
                    'display': 'flex',
                    'alignItems': 'center',
                    'zIndex': '1000',
                    'gap': '1px'
                })
            
            annotation_elements.append(element)
        
        return annotation_elements
    
    @app.callback(
        Output('text-annotations-store', 'data', allow_duplicate=True),
        [Input({'type': 'annotation-save', 'index': ALL}, 'n_clicks'),
         Input({'type': 'annotation-delete', 'index': ALL}, 'n_clicks'),
         Input({'type': 'annotation-text', 'index': ALL}, 'n_clicks')],
        [State({'type': 'annotation-input', 'index': ALL}, 'value'),
         State('text-annotations-store', 'data')],
        prevent_initial_call=True
    )
    def handle_annotation_actions(save_clicks, delete_clicks, text_clicks, input_values, annotations_data):
        """Handle save, delete, and edit actions for annotations."""
        
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        if not annotations_data:
            return no_update
        
        # Parse the triggered component
        import json
        try:
            component_info = json.loads(prop_id.split('.')[0])
            annotation_id = component_info['index']
            action_type = component_info['type']
        except (json.JSONDecodeError, KeyError, IndexError):
            return no_update
        
        if annotation_id not in annotations_data:
            return no_update
        
        annotations_data = dict(annotations_data)  # Make a copy
        
        if action_type == 'annotation-save':
            # Save the edited text
            if input_values:
                # Find the corresponding input value
                for i, (save_click, input_val) in enumerate(zip(save_clicks, input_values)):
                    if save_click and save_click > 0:
                        annotations_data[annotation_id]['text'] = input_val or 'New text'
                        annotations_data[annotation_id]['editing'] = False
                        break
        
        elif action_type == 'annotation-delete':
            # Delete the annotation
            del annotations_data[annotation_id]
        
        elif action_type == 'annotation-text':
            # Start editing existing text
            annotations_data[annotation_id]['editing'] = True
        
        return annotations_data
    
    # Add clientside callback for drag functionality
    app.clientside_callback(
        """
        function(children) {
            if (!children || children.length === 0) {
                return window.dash_clientside.no_update;
            }
            
            // Initialize drag functionality
            setTimeout(function() {
                let isDragging = false;
                let currentElement = null;
                let startX, startY, startLeft, startTop;
                
                // Remove existing event listeners
                document.querySelectorAll('.drag-handle').forEach(handle => {
                    if (handle._dragInitialized) return;
                    handle._dragInitialized = true;
                    
                    handle.addEventListener('mousedown', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        isDragging = true;
                        currentElement = e.target.closest('.draggable-annotation');
                        if (!currentElement) return;
                        
                        currentElement.classList.add('dragging');
                        
                        const rect = currentElement.getBoundingClientRect();
                        const containerRect = currentElement.offsetParent.getBoundingClientRect();
                        
                        startX = e.clientX;
                        startY = e.clientY;
                        
                        // Get current position as percentage
                        const currentLeft = parseFloat(currentElement.style.left) || 0;
                        const currentTop = parseFloat(currentElement.style.top) || 0;
                        startLeft = currentLeft;
                        startTop = currentTop;
                        
                        function handleMouseMove(e) {
                            if (!isDragging || !currentElement) return;
                            
                            const container = currentElement.offsetParent;
                            const containerRect = container.getBoundingClientRect();
                            
                            const deltaX = e.clientX - startX;
                            const deltaY = e.clientY - startY;
                            
                            const deltaXPercent = (deltaX / containerRect.width) * 100;
                            const deltaYPercent = (deltaY / containerRect.height) * 100;
                            
                            let newX = startLeft + deltaXPercent;
                            let newY = startTop + deltaYPercent;
                            
                            // Constrain to container bounds
                            newX = Math.max(0, Math.min(newX, 95));
                            newY = Math.max(0, Math.min(newY, 90));
                            
                            currentElement.style.left = newX + '%';
                            currentElement.style.top = newY + '%';
                        }
                        
                        function handleMouseUp(e) {
                            if (!isDragging) return;
                            
                            isDragging = false;
                            if (currentElement) {
                                currentElement.classList.remove('dragging');
                            }
                            currentElement = null;
                            
                            document.removeEventListener('mousemove', handleMouseMove);
                            document.removeEventListener('mouseup', handleMouseUp);
                        }
                        
                        document.addEventListener('mousemove', handleMouseMove);
                        document.addEventListener('mouseup', handleMouseUp);
                    });
                });
            }, 100);
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('text-annotations-container', 'style', allow_duplicate=True),
        Input('text-annotations-container', 'children'),
        prevent_initial_call=True
    )
    
