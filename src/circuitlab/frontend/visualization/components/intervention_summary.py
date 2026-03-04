from dash import html
from typing import List, Optional
from ...data.models import InterventionData


class InterventionSummary:
    """Component for displaying intervention analysis summary."""
    
    @staticmethod
    def create_summary_display(intervention_data_list: Optional[List[Optional[InterventionData]]], 
                              baseline_token: str) -> html.Div:
        """Create intervention summary display showing overall impact statistics."""
        
        if not intervention_data_list:
            return html.Div([
                html.Div("🎯 Intervention Summary", style={
                    'fontSize': '16px',
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
                'padding': '12px',
                'backgroundColor': '#f9fafb',
                'borderRadius': '6px',
                'border': '1px solid #e5e7eb'
            })
        
        # Filter out None entries and extract all intervention results
        valid_interventions = [data for data in intervention_data_list if data is not None]
        
        if not valid_interventions:
            return InterventionSummary.create_summary_display(None, baseline_token)
        
        # Flatten all intervention results from all features
        all_intervention_results = []
        for intervention_data in valid_interventions:
            all_intervention_results.extend(intervention_data.interventions)
        
        if not all_intervention_results:
            return InterventionSummary.create_summary_display(None, baseline_token)
        
        # Calculate summary statistics across all intervention results
        total_features = len(valid_interventions)
        total_interventions = len(all_intervention_results)
        negative_changes = sum(1 for result in all_intervention_results if result.baseline_prob_change < 0)
        positive_changes = sum(1 for result in all_intervention_results if result.baseline_prob_change > 0)
        
        avg_negative_change = sum(result.baseline_prob_change for result in all_intervention_results 
                                if result.baseline_prob_change < 0) / max(negative_changes, 1)
        
        avg_positive_change = sum(result.baseline_prob_change for result in all_intervention_results 
                                if result.baseline_prob_change > 0) / max(positive_changes, 1)
        
        # Find most impactful intervention results
        most_harmful = min(all_intervention_results, key=lambda x: x.baseline_prob_change)
        most_beneficial = max(all_intervention_results, key=lambda x: x.baseline_prob_change)
        
        # Get unique intervention values
        intervention_values = sorted(set(result.intervention_value for result in all_intervention_results))
        
        return html.Div([
            # Header
            html.Div([
                html.Span("🎯 Intervention Analysis Summary", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#374151'
                })
            ], style={'marginBottom': '12px'}),
            
            # Baseline token info and intervention values
            html.Div([
                html.Div([
                    html.Strong("Baseline Prediction: "),
                    html.Span(f'"{baseline_token}"', style={
                        'backgroundColor': '#e0e7ff',
                        'padding': '2px 6px',
                        'borderRadius': '4px',
                        'fontWeight': '500'
                    })
                ], style={'marginBottom': '6px'}),
                html.Div([
                    html.Strong("Intervention Values: "),
                    html.Span(f"{', '.join(map(str, intervention_values))}", style={
                        'backgroundColor': '#f3f4f6',
                        'padding': '2px 6px',
                        'borderRadius': '4px',
                        'fontSize': '11px',
                        'fontFamily': 'monospace'
                    })
                ])
            ], style={
                'fontSize': '13px',
                'marginBottom': '10px'
            }),
            
            # Summary statistics
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(str(total_features), style={
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'color': '#374151'
                        }),
                        html.Div("Active Features", style={
                            'fontSize': '11px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '8px',
                        'backgroundColor': '#f3f4f6',
                        'borderRadius': '6px',
                        'flex': '1'
                    }),
                    
                    html.Div([
                        html.Div(str(total_interventions), style={
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'color': '#6366f1'
                        }),
                        html.Div("Total Tests", style={
                            'fontSize': '11px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '8px',
                        'backgroundColor': '#eef2ff',
                        'borderRadius': '6px',
                        'flex': '1'
                    }),
                    
                    html.Div([
                        html.Div(str(negative_changes), style={
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'color': '#dc2626'
                        }),
                        html.Div("Harmful", style={
                            'fontSize': '11px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '8px',
                        'backgroundColor': '#fef2f2',
                        'borderRadius': '6px',
                        'flex': '1'
                    }),
                    
                    html.Div([
                        html.Div(str(positive_changes), style={
                            'fontSize': '20px',
                            'fontWeight': '700',
                            'color': '#16a34a'
                        }),
                        html.Div("Beneficial", style={
                            'fontSize': '11px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'textAlign': 'center',
                        'padding': '8px',
                        'backgroundColor': '#f0fdf4',
                        'borderRadius': '6px',
                        'flex': '1'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '8px',
                    'marginBottom': '12px'
                })
            ]),
            
            # Average changes
            html.Div([
                html.Div([
                    html.Strong("Avg. Impact: ", style={'fontSize': '12px'}),
                    html.Span(f"{avg_negative_change:+.3f}", style={
                        'fontSize': '12px',
                        'fontWeight': '600',
                        'color': '#dc2626',
                        'marginRight': '8px'
                    }),
                    html.Span(" / ", style={'fontSize': '12px', 'color': '#6b7280'}),
                    html.Span(f"{avg_positive_change:+.3f}", style={
                        'fontSize': '12px',
                        'fontWeight': '600',
                        'color': '#16a34a',
                        'marginLeft': '8px'
                    })
                ], style={'marginBottom': '8px'})
            ]) if negative_changes > 0 and positive_changes > 0 else None,
            
            # Most impactful interventions
            html.Div([
                html.Div([
                    html.Strong("Most Harmful: ", style={'fontSize': '12px'}),
                    html.Span(f"{most_harmful.baseline_prob_change:.3f}", style={
                        'fontSize': '12px',
                        'fontWeight': '600',
                        'color': '#dc2626',
                        'backgroundColor': '#fef2f2',
                        'padding': '2px 4px',
                        'borderRadius': '3px'
                    }),
                    html.Span(f" (×{most_harmful.intervention_value})", style={
                        'fontSize': '10px',
                        'color': '#6b7280',
                        'marginLeft': '4px'
                    })
                ], style={'marginBottom': '6px'}),
                
                html.Div([
                    html.Strong("Most Beneficial: ", style={'fontSize': '12px'}),
                    html.Span(f"{most_beneficial.baseline_prob_change:+.3f}", style={
                        'fontSize': '12px',
                        'fontWeight': '600',
                        'color': '#16a34a',
                        'backgroundColor': '#f0fdf4',
                        'padding': '2px 4px',
                        'borderRadius': '3px'
                    }),
                    html.Span(f" (×{most_beneficial.intervention_value})", style={
                        'fontSize': '10px',
                        'color': '#6b7280',
                        'marginLeft': '4px'
                    })
                ])
            ])
            
        ], style={
            'padding': '12px',
            'backgroundColor': '#f9fafb',
            'borderRadius': '6px',
            'border': '1px solid #e5e7eb',
            'fontFamily': 'Inter, system-ui, sans-serif'
        })

    @staticmethod
    def create_compact_intervention_indicator(intervention_data: Optional[InterventionData], selected_strength: float = -10.0) -> html.Span:
        """Create a compact indicator showing intervention impact for individual features."""
        
        if not intervention_data or not intervention_data.interventions:
            return html.Span()
        
        # Find the intervention result matching the selected strength
        selected_result = None
        for result in intervention_data.interventions:
            if abs(result.intervention_value - selected_strength) < 0.001:
                selected_result = result
                break
        
        if not selected_result:
            return html.Span()
            
        change = selected_result.baseline_prob_change
        
        if abs(change) < 0.001:
            color = '#6b7280'
            icon = '➖'
            text = "neutral"
        elif change < -0.01:
            color = '#dc2626'
            icon = '🔴'
            text = "high impact"
        elif change < -0.001:
            color = '#f59e0b'
            icon = '🟡'
            text = "low impact"
        elif change > 0.01:
            color = '#16a34a'
            icon = '🟢'
            text = "beneficial"
        else:
            color = '#10b981'
            icon = '🟢'
            text = "slight benefit"
        
        return html.Span([
            icon,
            " ",
            html.Span(f"{change:+.3f}", style={
                'fontWeight': '600',
                'fontSize': '11px'
            })
        ], 
        title=f"Intervention {text}: {change:+.4f} change at strength ×{selected_strength}",
        style={
            'fontSize': '10px',
            'color': color,
            'backgroundColor': f'{color}15',
            'padding': '1px 4px',
            'borderRadius': '3px',
            'marginLeft': '4px',
            'display': 'inline-block'
        })
    
    
    @staticmethod
    def create_simple_intervention_display(intervention_data: Optional[InterventionData], selected_strength: float = -10.0) -> html.Div:
        """Create lightweight single-line intervention display - now static (no interactive buttons)."""
        
        if not intervention_data or not intervention_data.interventions:
            return html.Div("No intervention data available", style={
                'fontSize': '11px',
                'color': '#9ca3af',
                'fontStyle': 'italic'
            })
        
        # Find the intervention result matching the selected strength
        selected_result = None
        for result in intervention_data.interventions:
            if abs(result.intervention_value - selected_strength) < 0.001:
                selected_result = result
                break
        
        if not selected_result:
            return html.Div(f"No data for strength ×{selected_strength}", style={
                'fontSize': '11px',
                'color': '#9ca3af',
                'fontStyle': 'italic'
            })
        
        # Create compact info for left side
        change_color = '#dc2626' if selected_result.baseline_prob_change < 0 else '#16a34a'
        change_text = f"{selected_result.baseline_prob_change:+.3f}"
        top_token = selected_result.tokens[0] if selected_result.tokens else 'N/A'
        
        return html.Div([
            html.Div([
                html.Span(f'"{selected_result.baseline_token}"', style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'marginRight': '6px'
                }),
                html.Span(f"({selected_result.baseline_prob_original:.3f})", style={
                    'fontSize': '10px',
                    'color': '#6b7280',
                    'marginRight': '4px'
                }),
                html.Span("→", style={
                    'fontSize': '10px',
                    'color': '#9ca3af',
                    'marginRight': '4px'
                }),
                html.Span(f'"{top_token}"', style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'marginRight': '6px'
                }),
                html.Span(change_text, style={
                    'fontSize': '11px',
                    'fontWeight': '600',
                    'color': change_color,
                    'backgroundColor': f'{change_color}10',
                    'padding': '1px 4px',
                    'borderRadius': '3px'
                }),
                html.Span(f" (×{selected_strength:g})", style={
                    'fontSize': '9px',
                    'color': '#6b7280',
                    'marginLeft': '4px',
                    'fontFamily': 'monospace'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center'
            })
        ], style={
            'padding': '3px 4px',
            'backgroundColor': '#fbfbfb',
            'borderRadius': '3px',
            'border': '1px solid #f1f1f1',
            'fontSize': '11px'
        })
    
    @staticmethod
    def create_detailed_intervention_display(intervention_data: Optional[InterventionData]) -> html.Div:
        """Create detailed display showing all intervention values and their impacts."""
        
        if not intervention_data or not intervention_data.interventions:
            return html.Div("No intervention data available")
        
        intervention_items = []
        for result in sorted(intervention_data.interventions, key=lambda x: x.intervention_value):
            change = result.baseline_prob_change
            
            # Color based on impact
            if change < -0.01:
                color = '#dc2626'
                bg_color = '#fef2f2'
            elif change < 0:
                color = '#f59e0b'
                bg_color = '#fffbeb'
            elif change > 0.01:
                color = '#16a34a'
                bg_color = '#f0fdf4'
            else:
                color = '#6b7280'
                bg_color = '#f9fafb'
            
            intervention_items.append(
                html.Div([
                    html.Span(f"×{result.intervention_value:g}", style={
                        'fontFamily': 'monospace',
                        'fontWeight': '600',
                        'fontSize': '11px',
                        'marginRight': '8px',
                        'minWidth': '40px',
                        'display': 'inline-block'
                    }),
                    html.Span(f"{change:+.4f}", style={
                        'fontWeight': '600',
                        'fontSize': '11px',
                        'color': color,
                        'backgroundColor': bg_color,
                        'padding': '2px 4px',
                        'borderRadius': '3px'
                    }),
                    html.Span(f" → {result.tokens[0] if result.tokens else 'N/A'}", style={
                        'fontSize': '10px',
                        'color': '#6b7280',
                        'marginLeft': '6px'
                    })
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'marginBottom': '4px',
                    'padding': '3px 6px',
                    'borderRadius': '4px',
                    'backgroundColor': '#fafafa'
                })
            )
        
        return html.Div([
            html.Div("Intervention Results:", style={
                'fontSize': '12px',
                'fontWeight': '600',
                'marginBottom': '6px',
                'color': '#374151'
            }),
            html.Div(intervention_items, style={
                'maxHeight': '200px',
                'overflowY': 'auto'
            })
        ], style={
            'padding': '8px',
            'backgroundColor': '#f9fafb',
            'borderRadius': '6px',
            'border': '1px solid #e5e7eb',
            'fontSize': '11px'
        })
    
    @staticmethod
    def create_intervention_with_tokens_display(intervention_data: Optional[InterventionData]) -> html.Div:
        """Create intervention display showing top 4 tokens for each strength level."""
        
        if not intervention_data or not intervention_data.interventions:
            return html.Div()
        
        # Sort interventions by strength value
        sorted_interventions = sorted(intervention_data.interventions, key=lambda x: x.intervention_value)
        
        intervention_blocks = []
        for result in sorted_interventions:
            change = result.baseline_prob_change
            strength = result.intervention_value
            
            # Determine color based on impact
            if abs(change) < 0.001:
                color = '#9ca3af'
                bg_color = '#f9fafb'
            elif change < 0:
                color = '#dc2626'
                bg_color = '#fef2f2'
            else:
                color = '#16a34a'
                bg_color = '#f0fdf4'
            
            # Get top 4 tokens
            top_tokens = []
            for i, (token, prob) in enumerate(zip(result.tokens[:4], result.probabilities[:4])):
                top_tokens.append(
                    html.Span([
                        f'"{token}"',
                        html.Span(f" ({prob:.2f})", style={
                            'fontSize': '8px',
                            'color': '#6b7280'
                        })
                    ], style={
                        'backgroundColor': '#fef3c7' if i == 0 else '#f3f4f6',
                        'padding': '1px 3px',
                        'borderRadius': '2px',
                        'margin': '1px',
                        'display': 'inline-block',
                        'fontSize': '9px',
                        'fontWeight': '500' if i == 0 else '400'
                    })
                )
            
            # Create intervention block
            intervention_blocks.append(
                html.Div([
                    # Header with strength and change
                    html.Div([
                        html.Span(f"×{strength:g}", style={
                            'fontFamily': 'monospace',
                            'fontWeight': '700',
                            'fontSize': '10px',
                            'marginRight': '4px'
                        }),
                        html.Span(f"{change:+.3f}", style={
                            'fontWeight': '600',
                            'fontSize': '10px',
                            'color': color
                        })
                    ], style={
                        'marginBottom': '1px',
                        'display': 'flex',
                        'alignItems': 'center'
                    }),
                    # Top tokens
                    html.Div(top_tokens, style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'gap': '1px'
                    })
                ], style={
                    'padding': '2px 4px',
                    'margin': '1px',
                    'backgroundColor': bg_color,
                    'borderRadius': '3px',
                    'border': f'1px solid {color}30',
                    'fontSize': '9px'
                })
            )
        
        return html.Div([
            html.Div(intervention_blocks, style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '1px'
            })
        ], style={
            'marginBottom': '6px',
            'padding': '4px',
            'backgroundColor': '#fafafa',
            'borderRadius': '3px',
            'border': '1px solid #e5e7eb'
        })
