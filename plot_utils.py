import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import pandas as pd
import seaborn as sns

from WA_params import out_path_0, metrics_of_interests, magnifying_factor, plot_n_cols, plot_n_rows

def read_data(weight):
    path = f'{out_path_0}/weight_{weight}/avg'
    filenames = os.listdir(path)
    metric_data_avg = {}
    for filename in filenames:
        k = filename[:-4]
        metric_data_avg[k] = pd.read_csv(f'{path}/{filename}')
        
    path = f'{out_path_0}/weight_{weight}/std'
    filenames = os.listdir(path)
    metric_data_std = {}
    for filename in filenames:
        k = filename[:-4]
        metric_data_std[k] = pd.read_csv(f'{path}/{filename}')

    return metric_data_avg, metric_data_std
    

def plotter(metric_dfs, plot_type='avg', colorbar_orientation="vertical"):
    # reset_defaults should ensure consistent behavior, 
    # but since all params are now explicit, there should be no need
    # sns.reset_defaults()
    sns.set_theme(font_scale=1.1)
    
    fig, axes = plt.subplots(plot_n_rows, plot_n_cols, figsize=(20, 13), sharex='col', sharey='row')  

    # corr_min = 0
    # custom_cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'green'])
    corr_min = -1
    custom_cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'green'])
    if plot_type == 'std':
        corr_min = 0
        custom_cmap = LinearSegmentedColormap.from_list('custom', ['white', 'green'])
    mappable = None  # will store the last heatmap for colorbar

    for idx_0, m in enumerate(metrics_of_interests):
        idx = (idx_0 // plot_n_cols, idx_0 % plot_n_cols)
        heatmap_data = metric_dfs[m].pivot(index='cost', columns='P', values='value')

        im = sns.heatmap(
            heatmap_data, annot=True, fmt='.0f', cmap=custom_cmap,
            ax=axes[idx], vmin=corr_min * magnifying_factor, vmax=magnifying_factor,
            cbar=False, xticklabels=True, yticklabels=True
        )
        mappable = im.collections[0]  # save the QuadMesh for colorbar

        axes[idx].set_title(f'{m.upper()}', fontsize=16)
        axes[idx].invert_yaxis()
        
        axes[idx].xaxis.tick_bottom()
        axes[idx].yaxis.tick_left()
        axes[idx].tick_params(axis='both', which='both', length=2, width=1)
        # axes[idx].xaxis.tick_top()
        # axes[idx].yaxis.tick_right()
        
        # axes[idx].tick_params(axis='y', labelrotation=90)

        # labels
        if idx[1] == 0:
            axes[idx].set_ylabel('$r_C$', fontdict={'size':16})
            # axes[idx].set_ylabel('$C_{FN}\,/\,(C_{FN}+C_{FP})$', fontdict={'size':16})
        else:
            axes[idx].set_ylabel('')
        axes[idx].set_xlabel('')  # will be set later for bottom-most visible plots

    # remove empty plots and fix x-axis labels
    n_used = len(metrics_of_interests)
    empty_axes = []
    for idx, ax in enumerate(axes.flat):
        if idx >= n_used:
            ax.set_visible(False)
            empty_axes.append(ax)
    
    # Add x-axis labels and ticks to bottom-most visible plots in each column
    for col in range(plot_n_cols):
        bottom_row_in_col = None
        for row in range(plot_n_rows-1, -1, -1):
            plot_idx = row * plot_n_cols + col
            if plot_idx < n_used:
                bottom_row_in_col = row
                break
        if bottom_row_in_col is not None:
            axes[bottom_row_in_col, col].set_xlabel('$r_+$', fontdict={'size':16})
            axes[bottom_row_in_col, col].tick_params(axis='x', labelbottom=True, labelrotation=90)

    #add colorbar
    if empty_axes and mappable is not None:
        cax = empty_axes[-1]  # choose the first free subplot
        # cax.set_visible(True)
        # cax.clear()
        # cax.axis("off")

        # if colorbar_orientation == "horizontal":
        cb_ax = inset_axes(cax, width="60%", height="15%", loc="center")
        fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")
        # else:  # vertical
        #     cb_ax = inset_axes(cax, width="10%", height="100%", loc="center")
        #     fig.colorbar(mappable, cax=cb_ax, orientation="vertical")

        # fig.colorbar(mappable, ax=cax, orientation=colorbar_orientation)

    plt.subplots_adjust(hspace=0.23, wspace=0.035)
    # plt.tight_layout()

    plt.show()
 