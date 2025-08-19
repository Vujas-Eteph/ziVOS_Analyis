"""
Custom color map for the streamlit application.

by Stéphane Vujasinovic
"""
# - IMPORTS ---
from st_utils.ColorBlindPalette import hex_to_rgba, susielu_colors, basic_colors


# - CUSTOM COLOR MAP ---
def get_color_dict():
    """
    Create my color dict
    """
    return {
        'IoU':               f"rgba(4,104,138,1)",
        'Total_H_base':      f"rgba{hex_to_rgba(basic_colors['Red'], alpha=1.0)}",
        'Total_H_masked_gt': f"rgba{hex_to_rgba(basic_colors['Red'], alpha=1.0)}",
        'Total_H_masked_pd': f"rgba(224,110,18,1)",

        # TP Family with clearer variation
        'TP_H_base':        f"rgba{hex_to_rgba(susielu_colors['Gold'], alpha=1.0)}",
        'TP_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Gold'], alpha=1.0)}",
        'TP_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Gold'], alpha=1.0)}",

        # TN Family with clearer variation
        'TN_H_base':        f"rgba{hex_to_rgba(susielu_colors['Peach'], alpha=1.0)}",
        'TN_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Peach'], alpha=1.0)}",
        'TN_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Peach'], alpha=1.0)}",

        # FP Family with clearer variation
        'FP_H_base':        f"rgba{hex_to_rgba(susielu_colors['Raspberry'], alpha=1.0)}",
        'FP_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Raspberry'], alpha=1.0)}",
        'FP_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Raspberry'], alpha=1.0)}",

        # FN Family with clearer variation
        'FN_H_base':        f"rgba{hex_to_rgba(susielu_colors['Orchid'], alpha=1.0)}",
        'FN_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Orchid'], alpha=1.0)}",
        'FN_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Orchid'], alpha=1.0)}",

        # TR Family with clearer variation
        'TR_H_base':        f"rgba{hex_to_rgba(susielu_colors['Dark Orchid'], alpha=1.0)}",
        'TR_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Dark Orchid'], alpha=1.0)}",
        'TR_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Dark Orchid'], alpha=1.0)}",

        # FR Family with clearer variation
        'FR_H_base':        f"rgba{hex_to_rgba(susielu_colors['Blue'], alpha=1.0)}",
        'FR_H_masked_gt':   f"rgba{hex_to_rgba(susielu_colors['Blue'], alpha=1.0)}",
        'FR_H_masked_pd':   f"rgba{hex_to_rgba(susielu_colors['Blue'], alpha=1.0)}",

        'gt_object_size':   f"rgba{hex_to_rgba(basic_colors['Green'], alpha=1.0)}",
        'pd_object_size':   f"rgba{hex_to_rgba(basic_colors['Red'], alpha=1.0)}",

        'Obj_in_GT_flag':   f"rgba{hex_to_rgba(basic_colors['Green'], alpha=0.7)}",
        'Obj_in_PD_flag':   f"rgba{hex_to_rgba(basic_colors['Red'], alpha=0.7)}",
        'Failure':          f"rgba{hex_to_rgba(basic_colors['Blue'], alpha=0.7)}",

        'TP_size':          f"rgba{hex_to_rgba(susielu_colors['Gold'], alpha=1.0)}",
        'TN_size':          f"rgba{hex_to_rgba(susielu_colors['Peach'], alpha=1.0)}",
        'FP_size':          f"rgba{hex_to_rgba(susielu_colors['Raspberry'], alpha=1.0)}",
        'FN_size':          f"rgba{hex_to_rgba(susielu_colors['Orchid'], alpha=1.0)}",
        'TR_size':          f"rgba{hex_to_rgba(susielu_colors['Dark Orchid'], alpha=1.0)}",
        'FR_size':          f"rgba{hex_to_rgba(susielu_colors['Blue'], alpha=1.0)}",
        }
