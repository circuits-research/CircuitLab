from dash import html
import math
from typing import Dict, Any, Optional

class LanguageDisplay:
    """Component for displaying multilingual feature analysis."""
    
    # Language mapping from full names to 2-letter codes
    LANGUAGE_MAPPING = {
        'fra_Latn': 'fr',
        'eng': 'en', 
        'deu_Latn': 'ge',
        'arb_Arab': 'ar',
        'cmn_Hani': 'ch'
    }
    
    # Color mapping for languages - lighter colors
    LANGUAGE_COLORS = {
        'en': '#86efac',  # light green
        'fr': '#93c5fd',  # light blue
        'ge': '#fdba74',  # light orange
        'ar': '#fca5a5',  # light red
        'ch': '#fde047'   # light yellow
    }
    
    @staticmethod
    def _calculate_entropy(distribution: Dict[str, float]) -> float:
        """Calculate normalized entropy of language distribution (0-1 scale)."""
        if not distribution or sum(distribution.values()) == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy (log2 of number of languages)
        num_languages = len([p for p in distribution.values() if p > 0])
        if num_languages <= 1:
            return 0.0
        
        max_entropy = math.log2(num_languages)
        normalized_entropy = entropy / max_entropy
        
        return min(normalized_entropy, 1.0)  # Ensure it doesn't exceed 1
    
    @staticmethod
    def _normalize_language_keys(distribution: Dict[str, float]) -> Dict[str, float]:
        """Convert language keys to 2-letter codes."""
        normalized = {}
        for lang, prob in distribution.items():
            short_lang = LanguageDisplay.LANGUAGE_MAPPING.get(lang, lang)
            normalized[short_lang] = prob
        return normalized
    
    @staticmethod
    def create_language_bars(distribution: Dict[str, float], title: str, entropy: float) -> html.Div:
        """Create compact horizontal bar chart for language distribution."""
        if not distribution:
            return html.Div([
                html.Div(title, style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginBottom': '3px'
                }),
                html.Div("No data", style={
                    'fontSize': '9px',
                    'color': '#6b7280',
                    'fontStyle': 'italic'
                })
            ])
        
        # Normalize and sort by probability
        normalized_dist = LanguageDisplay._normalize_language_keys(distribution)
        sorted_langs = sorted(normalized_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Color entropy based on threshold (normalized 0-1 scale)
        entropy_color = '#16a34a' if entropy > 0.8 else '#dc2626'  # green if high entropy (>0.8), red if low
        entropy_bg = '#dcfce7' if entropy > 0.8 else '#fef2f2'     # light green/red backgrounds
        
        bars = []
        for lang, prob in sorted_langs:
            if prob > 0:  # Show all non-zero probabilities
                color = LanguageDisplay.LANGUAGE_COLORS.get(lang, '#d1d5db')
                width_percent = prob * 100
                
                bars.append(
                    html.Div([
                        html.Div([
                            html.Span(lang.upper(), style={
                                'fontSize': '9px',
                                'fontWeight': '500',
                                'color': '#374151',
                                'marginRight': '4px'
                            }),
                            html.Span(f"{prob:.2f}", style={
                                'fontSize': '8px',
                                'color': '#6b7280'
                            })
                        ], style={
                            'backgroundColor': color,
                            'height': '14px',
                            'width': f'{max(width_percent, 25)}%',  # Increased minimum width
                            'display': 'flex',
                            'alignItems': 'center',
                            'paddingLeft': '4px',
                            'paddingRight': '2px',  # Add right padding
                            'borderRadius': '2px',
                            'minWidth': '50px',  # Increased minimum width
                            'overflow': 'hidden',  # Ensure text doesn't overflow
                            'whiteSpace': 'nowrap'  # Prevent text wrapping
                        })
                    ], style={
                        'marginBottom': '1px',
                        'display': 'flex',
                        'alignItems': 'center'
                    })
                )
        
        return html.Div([
            html.Div([
                html.Span(f"{title}:", style={
                    'fontSize': '11px',
                    'fontWeight': '500',
                    'color': '#374151',
                    'marginRight': '6px'
                }),
                html.Span(f"H={entropy:.2f}", style={
                    'fontSize': '9px',
                    'fontWeight': '600',
                    'color': entropy_color,
                    'backgroundColor': entropy_bg,
                    'padding': '1px 4px',
                    'borderRadius': '3px'
                })
            ], style={
                'marginBottom': '4px',
                'display': 'flex',
                'alignItems': 'center'
            }),
            html.Div(bars, style={
                'marginBottom': '2px'
            })
        ])
    
    @staticmethod
    def create_language_analysis(feature_config: Dict[str, Any]) -> Optional[html.Div]:
        """Create complete language analysis display for a feature."""
        # Check if multilingual data is available
        lang_dist = feature_config.get('language_distribution')
        general_lang_dist = feature_config.get('general_language_distribution')
        
        if not lang_dist and not general_lang_dist:
            return None
        
        components = []
        
        # Top sequences language distribution
        if lang_dist:
            entropy_top = LanguageDisplay._calculate_entropy(lang_dist)
            components.append(
                LanguageDisplay.create_language_bars(
                    lang_dist, 
                    "Top Sequences", 
                    entropy_top
                )
            )
        
        # General language distribution  
        if general_lang_dist:
            entropy_general = LanguageDisplay._calculate_entropy(general_lang_dist)
            components.append(
                LanguageDisplay.create_language_bars(
                    general_lang_dist,
                    "All Activations", 
                    entropy_general
                )
            )
        
        if not components:
            return None
        
        return html.Div([
            html.Div(components, style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gap': '8px'
            })
        ], style={
            'padding': '6px 8px',
            'backgroundColor': '#f3f4f6',
            'borderRadius': '4px',
            'marginBottom': '6px',
            'fontSize': '11px'
        })
