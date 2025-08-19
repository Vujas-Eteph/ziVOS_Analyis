"""
Custom color maps palette

Quick use - usage in another script:
    from color_blind_palette import [name of the palette]

by Stephane Vujasinovic
"""
# - VANILLA COLORS ------------------------------------------------------------
basic_colors = {
    "Red":"#ff0f0f",
    "Green":"#0fff0f",
    "Blue":"#0f0fff"
}
# -----------------------------------------------------------------------------

# - COLOR BLIND FRIENDLY ------------------------------------------------------
# Color map based on: https://projects.susielu.com/viz-palette
susielu_colors = {
    "Gold":"#ffd700",
    "Peach":"#ffb14e",
    "Salmon":"#fa8775",
    "Raspberry":"#ea5f94",
    "Orchid":"#cd34b5",
    "Dark Orchid":"#9d02d7",
    "Blue": "#0000ff"
}
# Color map based on: ???
ibm_colors = {
    "Ultramarine40": "#648fff",
    "Indigo50": "#785ef0",
    "Magenta50": "#dc267f",
    "Orange40": "#fe6100",
    "Gold20": "#ffb000",
    "Black100": "#000000",
    "White0": "#ffffff"
}
general_color_palette = {
    "VividCyan": "#01c5c4",
    "DeepSapphire": "#0a3d62",
    "FreshGreen": "#6ab04c",
    "SunflowerYellow": "#f9ca24",
    "CoralRed": "#ff6348",
    "HotPink": "#ff69b4",
    "RoyalPurple": "#5352ed",
    "DarkSlate": "#34495e",
    "SoftOrange": "#ff7e67",
    "MidnightBlue": "#130f40",
    "EarthyBrown": "#8b4513",
    "SlateGray": "#7f8c8d",
    "OffWhite": "#f1f2f6",
    "ClassicBlack": "#000000",
    "PureWhite": "#ffffff"
}
# -----------------------------------------------------------------------------

# - STYLE COLORS --------------------------------------------------------------
st_color_palette = {
    'Red': '#FF4B4B',
    'Green': '#4BFF4B',
    'Blue': '#4B4BFF'
}
# -----------------------------------------------------------------------------

# - FUNCTIONS ---
def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to RGBA format."""
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgba = rgba + (alpha,)
    return rgba
