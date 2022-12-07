# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:25:51 2022

@author: Simon Kern
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt

def make_fig(n_axs=30, n_bottom=0, no_ticks=False, suptitle='',
             xlabel='timepoint', ylabel='accuracy', figsize=None):
    """
    helper function to create a grid space with RxC rows and a
    large row with two axis on the bottom
    
    returns: fig, axs(size=(rows*columns)), ax_left_bottom, ax_right_bottom
    """
    
    COL_MULT = 10 # to accomodate also too large axis
    # some heuristic for finding optimal rows and columns
    for columns in [2, 4, 6, 8]:
        rows = np.ceil(n_axs/columns).astype(int)
        if columns>=rows:
            break
    assert columns*rows>=n_axs
    
    if isinstance(n_bottom, int):
        n_bottom = [1 for _ in range(n_bottom)]
    n_axsb = len(n_bottom)
    
    COL_MULT = 1
    if n_axsb>0:
        for COL_MULT in range(1, 12):
            if (columns*COL_MULT)%n_axsb==0:
                break
        if not (columns*COL_MULT)%n_axsb==0:
            warnings.warn(f'{columns} cols cannot be evenly divided by {n_axsb} bottom plots')
    fig =  plt.figure(dpi=75, constrained_layout=True, figsize=figsize)
    # assuming maximum 30 participants
    gs = fig.add_gridspec((rows+2*(n_axsb>0)), columns*COL_MULT) # two more for larger summary plots
    axs = []
    
    # first the individual plot axis for each participant
    for x in range(rows):
        for y in range(columns):
            ax = fig.add_subplot(gs[x, y*COL_MULT:(y+1)*COL_MULT])
            if no_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            axs.append(ax)
            
    fig.suptitle(suptitle)

    if n_axsb==0:
        return fig, axs
    
    # second the two graphs with all data combined/meaned     
    axs_bottom = []
    step = np.ceil(columns*COL_MULT//n_axsb).astype(int)
    for b, i in enumerate(range(0, columns*COL_MULT, step)):
        if n_bottom[b]==0: continue # do not draw* this plot
        ax_bottom = fig.add_subplot(gs[rows:, i:(i+step)])
        if xlabel: ax_bottom.set_xlabel(xlabel)
        if ylabel: ax_bottom.set_ylabel(ylabel)
        if i>0 and no_ticks: # remove yticks on righter plots
            ax_bottom.set_yticks([])
        axs_bottom.append(ax_bottom)
    for ax in axs[n_axs:]: ax.axis('off')
    return fig, axs, *axs_bottom