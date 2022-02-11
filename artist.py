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
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec


class TheArtist():


    def __init__(self, latex=True, font='serif', fontsize=8):
        
        self.inches_per_pt = 1.0 / 72.27   
        plt.rc('text', usetex=latex)
        plt.rc('font', family=font, size=fontsize)
        plt.rc('axes', titlesize=fontsize)

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

        elif type(ratio) == float:

            ratio = ratio
        
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

            elif type(rows) == list and len(rows) == cols:
            
                fig_width = fig_width_pt * self.inches_per_pt 
                fig_height = fig_width * max(rows) / cols * ratio

                self.fig = plt.figure(figsize = (fig_width, fig_height), constrained_layout=False)

                gs = GridSpec(max(rows) * 2, cols * 2, figure=self.fig)

                self.axs = []
                
                for row, col in zip(rows, range(cols)):

                    axs_row = []

                    if row == max(rows):

                        for idx_row in range(row):

                            axs_row.append(
                                    self.fig.add_subplot(gs[2*idx_row:2*(idx_row + 1), 2*col:2*(col + 1)])
                                )

                    else:

                        start = max(rows) - row

                        for idx_row in range(row):

                            axs_row.append(
                                self.fig.add_subplot(gs[(start + 2*idx_row):(start + 2*(idx_row + 1)), 2*col:2*(col + 1)])
                            )

                    self.axs.append(
                        axs_row
                    )
                        
                
                self.axs = np.array(
                    [xi + [None] * (max(map(len, self.axs)) - len(xi)) for xi in self.axs]
                ).T

        
        self.im = []

        return

    
    def plot_probability_density_function(self, data, bins, idx_row, idx_col, color='darkblue', edgecolor='black', linewidth=1, hist=False, kde=True):
        
        sns.distplot(
            data, 
            hist=hist, 
            kde=kde, 
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


    def plot_scatter(self, x, y, idx_row, idx_col, marker='s', edgecolor=None, facecolor=None, linewidths=3, markersize=4,zorder=1):

        self.axs[idx_row, idx_col].scatter(
            x,
            y,
            marker=marker,
            s=markersize,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidths=linewidths,
            zorder=zorder
        )

        return


    def plot_panel_imshow(self, z, idx_row, idx_col, vmin=0, vmax=1, origin='lower', extent=None, cmap='Reds'):

        self.im.append(
            self.axs[idx_row, idx_col].imshow(
            X=z,
            origin=origin,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        )

        return

    
    def plot_panel_contourf(self, x, y, z, idx_row, idx_col, cmap='Reds', clims=[-1,1], levels=10, extend='both', norm=matplotlib.colors.Normalize(vmin=-1, vmax=1)):
        
        self.im.append(
            self.axs[idx_row, idx_col].contourf(
                x,
                y,
                z,
                levels=levels,
                cmap=cmap,
                vmin=clims[0],
                vmax=clims[1],
                extend=extend,
                norm=norm
            )
        )

        return


    def plot_panel_contour(self, x, y, z, idx_row, idx_col, colors=None, cmap='Reds', clims=[-1,1], levels=10, extend='both', linewidths=1, linestyles='-'):
        
        self.im.append(
            self.axs[idx_row, idx_col].contour(
                x,
                y,
                z,
                levels=levels,
                colors=colors,
                cmap=cmap,
                linewidths=linewidths,
                linestyles=linestyles
                # vmin=clims[0],
                # vmax=clims[1],
                # extend=extend
            )
        )

        return


    def plot_hist(self, x, bins, idx_row, idx_col, color='darkblue', edgecolor=None, histtype='bar', density=False, weights=None, stacked=False):

        self.axs[idx_row, idx_col].hist(
            x=x,
            bins=bins,
            color=color,
            histtype=histtype,
            density=density,
            weights=weights,
            stacked=stacked,
            edgecolor=edgecolor
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


    def plot_lines_semi_x(self, x, y, idx_row, idx_col, color='darkblue', linewidth=1, linestyle='-', marker=None, markeredgecolor=None, markerfacecolor=None, markersize=2):

        self.axs[idx_row, idx_col].semilogx(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markersize=3
        )

        return


    def plot_errorbar(self, x, y, yerr, idx_row, idx_col, color='darkblue', zorder=1, capsize=3, linestyle='-'):

        self.axs[idx_row, idx_col].errorbar(
            x,
            y,
            yerr,
            linestyle=linestyle,
            color=color,
            zorder=zorder,
            capsize=capsize
        )

        return


    def plot_patch(self, polygon, idx_row, idx_col, alpha=0.3, edgecolor=None, facecolor='coral'):

        poly = Polygon(polygon, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)

        self.axs[idx_row, idx_col].add_patch(poly)

        return


    def plot_bar_horizontal(self, x, y, idx_row, idx_col, color='lightcoral', edgecolor='k'):

        self.axs[idx_row, idx_col].barh(
            y=x,
            width=y,
            color=color, 
            edgecolor=edgecolor
        )


    def save_figure(self, fig_name, fig_format='png', dots_per_inch=600):

        filename = f'{fig_name}.{fig_format}'

        # self.fig.tight_layout()

        self.fig.savefig(filename, dpi=dots_per_inch, bbox_inches='tight')

        return
    

    def set_axis_lims(self, values, idx_row, idx_col):

        self.axs[idx_row, idx_col].set_xlim(values[0])
        self.axs[idx_row, idx_col].set_ylim(values[1])

        return


    def set_colorbar(self, idx_fig, idx_row, idx_col, title=False, orientation='vertical', fraction=0.05, ticks=[-1, 0, 1], vmin=-1, vmax=1):

        cbar = self.fig.colorbar(
            self.im[idx_fig], 
            ax=self.axs[idx_row, idx_col], 
            fraction=fraction, 
            orientation=orientation, 
            ticks=ticks,
            extendfrac=0
        )

        if title:

            cbar.ax.set_ylabel(title, rotation=-90, va="bottom")

        return


    def set_labels(self, labels, idx_row, idx_col, labelpad=[None, None], flip = [False, False]):

        self.axs[idx_row, idx_col].set_xlabel(labels[0], labelpad=labelpad[0])
        self.axs[idx_row, idx_col].set_ylabel(labels[1], labelpad=labelpad[1])

        if flip[0]:

            self.axs[idx_row, idx_col].xaxis.set_label_position("top")

        if flip[1]:

            self.axs[idx_row, idx_col].yaxis.set_label_position("right")

        return


    def set_ticks(self, ticks, idx_row, idx_col):

        if ticks[0] != None:
        
            self.axs[idx_row, idx_col].set_xticks(ticks[0])

        if ticks[1] != None:

            self.axs[idx_row, idx_col].set_yticks(ticks[1])

        return


    def set_ticklabels(self, labels, idx_row, idx_col, position=[False, True, False, True], rotation=[0, 0], alignment_h=["center", "center"], alignment_v=["center", "center"], rotation_mode=[None, None]):

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


    def set_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):

        # plt.subplots_adjust(top=0.6)

        self.fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        return


    def set_tick_params(self, idx_row, idx_col, axis="both", direction="in", which="both", pad=4, bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, length=4):
        
        self.axs[idx_row,idx_col].tick_params(axis=axis, direction=direction, pad=pad, which=which, bottom=bottom, top=top, labelbottom=labelbottom, left=left, right=right, labelleft=labelleft, length=length)

        return


    def set_scale_format(self, idx_row, idx_col, format_scale):

        self.axs[idx_row, idx_col].set_xscale(format_scale[0]) 
        self.axs[idx_row, idx_col].set_yscale(format_scale[1]) 
