# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:58:27 2020
@author: aguemes
"""

import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TheArtist():


    def __init__(self, latex=True, font='serif', fontsize=8):
        
        self.inches_per_pt = 1.0 / 72.27   
        plt.rc('text', usetex=latex)
        plt.rc('font', family=font, size=fontsize)

        return

    
    def draw(self, plot_dict):

        for plot in plot_dict.keys():

            if plot_dict[plot]['plot_type'] == 'pdf':
                
                self.plot_probability_density_function(plot_dict[plot]['data'], plot_dict[plot]['range'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])
            
            elif plot_dict[plot]['plot_type'] == 'cdf':
                
                self.plot_cumulative_density_function(plot_dict[plot]['data'], plot_dict[plot]['range'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])

            elif plot_dict[plot]['plot_type'] == 'jpdf':
                
                self.plot_joint_probability_density_function(plot_dict[plot]['data'], plot_dict[plot]['range'][0], plot_dict[plot]['range'][1], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])

            else: 

                print('Plotting method not implemented yet. Exiting...')
                sys.exit(1)



            self.set_axis_lims(plot_dict[plot]['lims'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])
            self.set_ticks(plot_dict[plot]['ticks'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])
            self.set_labels(plot_dict[plot]['labels'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])
            self.set_title(plot_dict[plot]['title'], plot_dict[plot]['idx_row'], plot_dict[plot]['idx_col'])

        return


    def generate_figure_environment(self, cols=1, rows=1, fig_width_pt=384, ratio='golden', regular=True):

        if ratio == 'golden':

            ratio = 2 / (1 + 5 ** 0.5) 

        elif ratio == 'square':

            ratio = 1
        
        else:

            print('This ratio has not been defined yet. Exiting code...')
            
            sys.exit(1)


        if regular:
                       
            fig_width = fig_width_pt * self.inches_per_pt 
            fig_height = fig_width * rows / cols * ratio

            self.fig, self.axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
            
        else:

            if type(cols) == list and len(cols) == rows:
            
                fig_width = fig_width_pt * self.inches_per_pt 
                fig_height = fig_width * rows / max(cols) * ratio

                self.fig = plt.figure(figsize = (fig_width, fig_height), constrained_layout=False)

                gs = GridSpec(rows * 2, max(cols) * 2, figure=self.fig)

                self.axs = []
                
                for row, col in zip(range(rows), cols):

                    axs_row = []

                    if col == max(cols):

                        for idx_col in range(col):

                            axs_row.append(
                                    self.fig.add_subplot(gs[2*row:2*(row + 1), 2*idx_col:2*(idx_col + 1)])
                                )

                    else:

                        start = max(cols) - col

                        for idx_col in range(col):

                            axs_row.append(
                                self.fig.add_subplot(gs[2*row:2*(row + 1), (start + 2*idx_col):(start + 2*(idx_col + 1))])
                            )

                    self.axs.append(
                        axs_row
                    )
                        
                
                self.axs = np.array(
                    [xi + [None] * (max(map(len, self.axs)) - len(xi)) for xi in self.axs]
                )

        
        self.im = []

        return

    
    def plot_probability_density_function(self, data, bins, idx_row, idx_col, color='darkblue', edgecolor='black', linewidth=1):
        
        sns.distplot(
            data, 
            hist=True, 
            kde=True, 
            bins=bins, 
            color=color, 
            hist_kws={'edgecolor': edgecolor},
            kde_kws={'linewidth': linewidth, 'bw': (bins[1] - bins[0]) / 2},
            ax=self.axs[idx_row, idx_col]
        )

        return

    
    def plot_cumulative_density_function(self, data, bins, idx_row, idx_col, color='darkblue', edgecolor='black', linewidth=1):
        
        sns.distplot(
            data, 
            hist=False,  
            bins=bins, 
            color=color, 
            hist_kws={'edgecolor': edgecolor},
            kde_kws={'linewidth': linewidth, 'bw': (bins[1] - bins[0]) / 2, 'cumulative':True},
            ax=self.axs[idx_row, idx_col]
        )

        return

    
    def plot_joint_probability_density_function(self, data, xbins, ybins, idx_row, idx_col, levels=[0.1, 0.5, 0.9], color='darkblue', edgecolor='black', linewidth=1):
        

        jointProbs, xedges, yedges = np.histogram2d(x=data[0], y=data[1], bins=(xbins, ybins))
        jointProbs /= jointProbs.max()

        xedges = (xedges[1:] + xedges[:-1]) / 2
        yedges = (yedges[1:] + yedges[:-1]) / 2

        self.axs[idx_row, idx_col].contour(
            yedges,
            xedges,
            jointProbs,
            levels=levels,
            colors=color,
            linewidths=linewidth
        )

        return


    def plot_lines(self, x, y, idx_row, idx_col, color='darkblue', linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=3):

        self.axs[idx_row, idx_col].plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markersize=markersize
        )


    def plot_panel_imshow(self, x, y, z, idx_row, idx_col, origin='lower', extent=None, cmap='Reds'):

        # self.axs[idx_row, idx_col].imshow(
        #     X=z,
        #     origin=origin,
        #     extent=None,
        #     cmap=cmap,
        #     vmin=0,
        #     vmax=z.max()
        # )

        self.im.append(
            self.axs[idx_row, idx_col].imshow(
            X=z,
            origin=origin,
            extent=None,
            cmap=cmap,
            vmin=0,
            vmax=z.max()
        )
        )

        return


    def plot_hist(self, x, bins, idx_row, idx_col, color='darkblue', histtype='bar', density=False, weights=None):

        self.axs[idx_row, idx_col].hist(
            x=x,
            bins=bins,
            color=color,
            histtype=histtype,
            density=density,
            weights=weights
        )


    def plot_lines_semi_y(self, x, y, idx_row, idx_col, color='darkblue', linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None):

        self.axs[idx_row, idx_col].semilogy(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor
        )

    
    def save_figure(self, fig_name, fig_format='png', dots_per_inch=600):

        filename = f'{fig_name}.{fig_format}'

        self.fig.tight_layout()

        self.fig.savefig(filename, dpi=dots_per_inch, bbox_inches='tight')

        return
    

    def set_axis_lims(self, values, idx_row, idx_col):

        self.axs[idx_row, idx_col].set_xlim(values[0])
        self.axs[idx_row, idx_col].set_ylim(values[1])

        return


    def set_colorbar(self, idx_row, idx_col, title=False):

        cbar = self.fig.colorbar(self.im[0], ax=self.axs[idx_row, idx_col], fraction=0.04)

        if title:

            cbar.ax.set_ylabel(title, rotation=-90, va="bottom")

        return


    def set_labels(self, labels, idx_row, idx_col):

        self.axs[idx_row, idx_col].set_xlabel(labels[0])
        self.axs[idx_row, idx_col].set_ylabel(labels[1])

        return


    def set_ticks(self, ticks, idx_row, idx_col):

        if list(ticks[0]) != None:
        
            self.axs[idx_row, idx_col].set_xticks(ticks[0])

        if list(ticks[1]) != None:

            self.axs[idx_row, idx_col].set_yticks(ticks[1])

        return


    def set_ticklabels(self, labels, idx_row, idx_col, position=[True, False, True, False], rotation=[0, 0], alignment_h=["center", "center"], alignment_v=["center", "center"], rotation_mode=[None, None]):

        if labels[0] != None:
        
            self.axs[idx_row, idx_col].set_xticklabels(labels[0], rotation=rotation[0], ha=alignment_h[0], va=alignment_v[0], rotation_mode=rotation_mode[0])

        if labels[1] != None:

            self.axs[idx_row, idx_col].set_yticklabels(labels[1], rotation=rotation[1], ha=alignment_h[1], va=alignment_v[1], rotation_mode=rotation_mode[1])

        self.axs[idx_row, idx_col].tick_params(top=position[0], bottom=position[1], labeltop=position[2], labelbottom=position[3])


        # plt.setp(cezanne.axs[0,0].get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        return


    def set_title(self, title, idx_row, idx_col):

        self.axs[idx_row, idx_col].set_title(title)

        return

    
    def set_text(self, text, xpos, ypos, idx_row, idx_col):

        self.axs[idx_row, idx_col].text(xpos, ypos, text, transform=self.axs[idx_row,idx_col].transAxes)  
        
        return


