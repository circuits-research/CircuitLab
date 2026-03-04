from dash import html
from typing import Optional, Dict, Any
from ...data.models import InterventionData
from .intervention_summary import InterventionSummary
from .language_display import LanguageDisplay

class FeatureDisplay:
    """Component for displaying feature activation examples."""
    
    @staticmethod
    def create_intervention_display(intervention_data: Optional[InterventionData]) -> html.Div:
        """Create intervention analysis display component."""
        if not intervention_data:
            return html.Div([
                html.Div("🎯 Intervention Analysis", style={
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'color': '#374151',
                    'marginBottom': '8px'
                }),
                html.Div("No intervention data available", style={
                    'fontSize': '12px',
                    'color': '#6b7280',
                    'fontStyle': 'italic'
                })
            ], style={
                'padding': '10px',
                'backgroundColor': '#f9fafb',
                'borderRadius': '4px',
                'border': '1px solid #e5e7eb',
                'marginBottom': '8px'
            })
        
        # Create probability change indicator
        change_color = '#dc2626' if intervention_data.baseline_prob_change < 0 else '#16a34a'
        change_icon = '📉' if intervention_data.baseline_prob_change < 0 else '📈'
        change_text = f"{intervention_data.baseline_prob_change:+.3f}"
        
        # Create top tokens after intervention
        intervention_tokens = []
        for i, (token, prob) in enumerate(zip(intervention_data.tokens, intervention_data.probabilities)):
            # Highlight if this is the new top token
            is_top = i == 0
            intervention_tokens.append(
                html.Span([
                    html.Span(f"#{i+1}", style={
                        'fontSize': '10px',
                        'color': '#6b7280',
                        'marginRight': '2px'
                    }),
                    html.Span(f'"{token}"', style={
                        'fontWeight': '600' if is_top else '500',
                        'color': '#1f2937' if is_top else '#374151'
                    }),
                    html.Span(f" ({prob:.3f})", style={
                        'fontSize': '10px',
                        'color': '#6b7280'
                    })
                ], style={
                    'backgroundColor': '#fef3c7' if is_top else '#f3f4f6',
                    'padding': '2px 6px',
                    'borderRadius': '4px',
                    'margin': '2px',
                    'display': 'inline-block',
                    'fontSize': '12px'
                })
            )
        
        return html.Div([
            # Header
            html.Div([
                html.Span("🎯 Intervention Analysis", style={
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'color': '#374151',
                    'marginRight': '8px'
                }),
                html.Span([
                    change_icon,
                    " ",
                    change_text
                ], style={
                    'fontSize': '12px',
                    'fontWeight': '600',
                    'color': change_color,
                    'backgroundColor': f'{change_color}15',
                    'padding': '2px 6px',
                    'borderRadius': '4px'
                })
            ], style={
                'marginBottom': '8px',
                'display': 'flex',
                'alignItems': 'center'
            }),
            
            # Baseline information
            html.Div([
                html.Div([
                    html.Strong("Baseline: "),
                    f'"{intervention_data.baseline_token}" ',
                    html.Span(f"({intervention_data.baseline_prob_original:.3f})", style={
                        'color': '#6b7280',
                        'fontSize': '11px'
                    }),
                    " → ",
                    html.Span(f"({intervention_data.baseline_prob_after_intervention:.3f})", style={
                        'color': change_color,
                        'fontSize': '11px',
                        'fontWeight': '600'
                    })
                ], style={
                    'fontSize': '12px',
                    'marginBottom': '6px'
                })
            ]),
            
            # Top tokens after intervention
            html.Div([
                html.Strong("Top tokens after intervention:", style={'fontSize': '12px', 'marginBottom': '4px', 'display': 'block'}),
                html.Div(intervention_tokens, style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '2px'
                })
            ])
        ], style={
            'padding': '10px',
            'backgroundColor': '#f9fafb',
            'borderRadius': '4px',
            'border': '1px solid #e5e7eb',
            'marginBottom': '8px',
            'fontFamily': 'Inter, system-ui, sans-serif'
        })

    @staticmethod
    def create_activation_display(feature_config: Optional[Dict[str, Any]], 
                                 feature_info: Optional[Dict[str, Any]],
                                 intervention_data: Optional[InterventionData] = None) -> html.Div:
        """Create the feature activation display component for narrower panel."""
        if not feature_config or 'top_examples' not in feature_config:
            return html.Div(
                [
                    html.Div("🎯 Feature Analysis", style={
                        'fontSize': '16px',
                        'fontWeight': '600',
                        'color': '#1e293b',
                        'marginBottom': '8px'
                    }),
                ],
                style={
                    'textAlign': 'center',
                    'padding': '18px',
                    'backgroundColor': '#f9fafb',
                    'borderRadius': '6px',
                    'margin': '8px'
                }
            )
        
        # Top activating tokens - compact format
        top_tokens_spans = []
        top_tokens = feature_config.get('top_activating_tokens', [])
        if top_tokens:
            for i, token_info in enumerate(top_tokens):
                token = token_info.get('token', 'N/A')
                # freq = token_info.get('frequency', 0)
                
                # Different colors for top 3
                colors = [
                    {'bg': '#fef3c7', 'text': '#92400e'},  # Gold for #1
                    {'bg': '#e5e7eb', 'text': '#374151'},  # Silver for #2
                    {'bg': '#fed7aa', 'text': '#9a3412'}   # Bronze for #3
                ]
                color = colors[i] if i < len(colors) else colors[-1]
                
                top_tokens_spans.append(
                    html.Span(f'"{token}"', style={
                        'backgroundColor': color['bg'],
                        'color': color['text'],
                        'padding': '2px 6px',
                        'borderRadius': '4px',
                        'fontSize': '12px',
                        'fontWeight': '500',
                        'marginRight': '6px'
                    })
                )
        
        # Create all examples in one continuous list
        all_examples = []
        
        for sentence in feature_config['top_examples']:
            # Process entire sentence for highlighting, not just word by word
            import re
            
            # Split by highlighting markers while preserving them
            parts = re.split(r'(<<.*?>>)', sentence)
            
            colored_words = []
            for part in parts:
                if part.startswith("<<") and part.endswith(">>"):
                    # This is a highlighted section
                    clean_part = part[2:-2]  # Remove << and >>
                    colored_words.append(
                        html.Span(clean_part, style={
                            'backgroundColor': '#fef3c7',
                            'color': '#92400e',
                            'padding': '2px 4px',
                            'margin': '1px',
                            'borderRadius': '3px',
                            'fontWeight': '500',
                            'display': 'inline-block',
                            'fontSize': '12px',
                            'fontFamily': 'Inter, system-ui, sans-serif'
                        })
                    )
                elif part.strip():  # Non-empty regular part
                    # Split regular parts by spaces to create individual word spans
                    words = part.split()
                    for word in words:
                        if word.strip():  # Skip empty strings
                            colored_words.append(
                                html.Span(word, style={
                                    'backgroundColor': '#f8fafc',
                                    'color': '#64748b',
                                    'padding': '1px 3px',
                                    'margin': '1px',
                                    'borderRadius': '2px',
                                    'display': 'inline-block',
                                    'fontSize': '12px',
                                    'fontFamily': 'Inter, system-ui, sans-serif'
                                })
                            )
            
            # Add each sentence as a simple div with minimal spacing
            all_examples.append(
                html.Div(colored_words, style={
                    'lineHeight': '1.5',
                    'marginBottom': '6px'
                })
            )
        
        components = [
            # Single line with short description, Layer, Feature ID, Top Tokens
            html.Div([
                html.Span([
                    html.Span("📊 ", style={'marginRight': '4px'}),
                    feature_config['description']  # Short description (3 words max)
                ], style={
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'color': '#1f2937',
                    'marginRight': '10px'
                }),
                html.Span(f"L{feature_info['layer']}", style={
                    'backgroundColor': '#3b82f6',
                    'color': 'white',
                    'padding': '2px 6px',
                    'borderRadius': '4px',
                    'fontSize': '10px',
                    'fontWeight': '500',
                    'marginRight': '6px'
                }),
                html.Span(f"#{feature_info['feature_idx']}", style={
                    'backgroundColor': '#e5e7eb',
                    'color': '#4b5563',
                    'padding': '2px 6px',
                    'borderRadius': '4px',
                    'fontSize': '10px',
                    'fontWeight': '500',
                    'marginRight': '8px'
                }),
                *top_tokens_spans
            ], style={
                'marginBottom': '8px',
                'lineHeight': '1.5',
                'display': 'flex',
                'flexWrap': 'wrap',
                'alignItems': 'center'
            })
        ]
        
        # Add language analysis if available
        language_analysis = LanguageDisplay.create_language_analysis(feature_config)
        if language_analysis:
            components.append(language_analysis)
            
        # Add intervention analysis if available (handle new multiple interventions format)
        if intervention_data and intervention_data.interventions:
            # Add intervention display with top 4 tokens for each level
            components.append(InterventionSummary.create_intervention_with_tokens_display(intervention_data))
            
        # Add explanation section (longer description) 
        if feature_config.get('explanation'):
            components.append(html.Div([
                html.Div(feature_config.get('explanation', ''), style={
                    'fontSize': '12px',
                    'color': '#374151',
                    'lineHeight': '1.8',
                    'padding': '6px 8px',
                    'backgroundColor': '#f3f4f6',
                    'borderLeft': '3px solid #6b7280',
                    'borderRadius': '4px',
                    'marginBottom': '8px'
                })
            ]))
            
        # Add examples section - adjust height based on other components
        has_language_data = language_analysis is not None
        if intervention_data and has_language_data:
            max_height = '250px'  # Reduced when both intervention and language data present
        elif intervention_data or has_language_data:
            max_height = '300px'  # Medium height when one additional component present
        else:
            max_height = '400px'  # Full height when just examples
        components.append(html.Div(all_examples, style={
            'maxHeight': max_height,
            'overflowY': 'auto',
            'padding': '8px',
            'backgroundColor': '#fafafa',
            'borderRadius': '4px',
            'border': '1px solid #e5e7eb'
        }))
        
        return html.Div(components, style={
            'padding': '10px',
            'backgroundColor': '#f9fafb',
            'borderRadius': '4px',
            'margin': '8px',
            'fontFamily': 'Inter, system-ui, sans-serif',
            'fontSize': '12px'
        })
