import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import pandas as pd
import seaborn as sns

from WA_params import out_path_0, metrics_of_interests, magnifying_factor, plot_n_cols, plot_n_rows

def read_data(weight, use_case):
    path = f'{out_path_0}/{use_case}/weight_{weight}/avg'
    filenames = os.listdir(path)
    metric_data_avg = {}
    for filename in filenames:
        k = filename[:-4]
        metric_data_avg[k] = pd.read_csv(f'{path}/{filename}')
        
    path = f'{out_path_0}/{use_case}/weight_{weight}/std'
    filenames = os.listdir(path)
    metric_data_std = {}
    for filename in filenames:
        k = filename[:-4]
        metric_data_std[k] = pd.read_csv(f'{path}/{filename}')

    return metric_data_avg, metric_data_std
    

def plotter(metric_dfs, plot_type='avg'):
    sns.set_theme(font_scale=1.1)
    
    fig, axes = plt.subplots(plot_n_rows, plot_n_cols, figsize=(20, 13), sharex='col', sharey='row')  

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

        cb_ax = inset_axes(cax, width="60%", height="15%", loc="center")
        fig.colorbar(mappable, cax=cb_ax, orientation="horizontal")

    plt.subplots_adjust(hspace=0.23, wspace=0.035)

    plt.show()


from WA_params import n_mcfl, n_rfl, massive_customers_fraction_lst, revenue_fraction_lst

def plotter_churn_extreme(metric_dfs, plot_type='avg', colorbar_in_last_subplot=True):

    sns.set_theme(font_scale=1.1)

    fig, axes = plt.subplots(n_mcfl, n_rfl, figsize=(22, 14), sharex='col', sharey='row')

    corr_min = -1
    custom_cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'green'])
    if plot_type == 'std':
        corr_min = 0
        custom_cmap = LinearSegmentedColormap.from_list('custom', ['white', 'green'])
    mappable = None
    empty_axes = []

    for i in range(n_mcfl):
        for j in list(range(n_rfl))[::-1]:
            ax = axes[i, n_rfl - 1 - j]
            if colorbar_in_last_subplot and i == n_mcfl - 1 and j == 0:
                ax.set_visible(False)
                empty_axes.append(ax)
                break
            heatmap_data = metric_dfs[(i, j)].pivot(index='cost', columns='P', values='value')

            im = sns.heatmap(
                heatmap_data, annot=True, fmt='.0f', cmap=custom_cmap,
                ax=ax, vmin=corr_min * magnifying_factor, vmax=magnifying_factor,
                cbar=False, xticklabels=True, yticklabels=True
            )
            mappable = im.collections[0]

            # ax.set_title(f'mcf={massive_customers_fraction_lst[i]}, rf={revenue_fraction_lst[j]}', fontsize=11)
            ax.invert_yaxis()
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()
            ax.tick_params(axis='both', which='both', length=2, width=1)

            ax.set_ylabel('$r_C$' if j == n_rfl - 1 else '', fontdict={'size': 16})
            ax.set_xlabel('$r_+$' if i == n_mcfl - 1 else '', fontdict={'size': 16})
            if i == n_mcfl - 1:
                ax.tick_params(axis='x', labelbottom=True, labelrotation=90)

    if mappable is not None:
        if empty_axes:
            cax = empty_axes[-1]
            cax.set_visible(True)
            cax.axis('off')
            cb_ax = inset_axes(cax, width='60%', height='15%', loc='center')
            fig.colorbar(mappable, cax=cb_ax, orientation='horizontal')
        else:
            fig.colorbar(mappable, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.subplots_adjust(hspace=0.05, wspace=0.03, left=0.02, top=0.98)

    # row/col reference labels via annotations anchored to axes fraction + offset points
    for j in list(range(n_rfl))[::-1]:
        axes[0, n_rfl - 1 - j].annotate(
            '$f_{\\mathrm{r}}$=' + str(revenue_fraction_lst[j]),
            xy=(0.5, 1), xycoords='axes fraction',
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=16
        )
    for i in range(n_mcfl):
        axes[i, 0].annotate(
            '$f_{\\mathrm{mc}}$=' + str(massive_customers_fraction_lst[i]),
            xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-63, 0), textcoords='offset points',
            ha='center', va='center', fontsize=16, rotation=90
        )

    plt.show()
