import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Union
from matplotlib.figure import Figure


def sns_styleset():
    '''Configure parameters for plotting'''
    sns.set_theme(context='paper',
                  style='white',
                  palette='muted'
                  )
    mpl.rcParams['figure.dpi']        = 300
    mpl.rcParams['grid.color']        = '.8'
    mpl.rcParams['axes.edgecolor']    = '.15'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['xtick.bottom']      = True
    mpl.rcParams['ytick.left']        = True
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.color']       = '.15'
    mpl.rcParams['ytick.color']       = '.15'
    mpl.rcParams['xtick.major.size']  = 12
    mpl.rcParams['ytick.major.size']  = 12
    mpl.rcParams['font.family']       = 'sans-serif'
    mpl.rcParams['font.sans-serif']   = ['Liberation Sans']
    mpl.rcParams['font.weight']       = 'regular'
    mpl.rcParams['axes.titlesize']    = 30
    mpl.rcParams['axes.labelsize']    = 30
    mpl.rcParams['axes.labelweight'] = 'regular'
    mpl.rcParams['legend.fontsize']   = 30
    mpl.rcParams['legend.frameon']    = False
    mpl.rcParams['xtick.labelsize']   = 30
    mpl.rcParams['ytick.labelsize']   = 30
    # Boxplot elements
    mpl.rcParams['boxplot.medianprops.color'] = 'black'
    mpl.rcParams['boxplot.boxprops.linewidth'] = 2
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 2
    mpl.rcParams['boxplot.capprops.linewidth'] = 2
    mpl.rcParams['boxplot.medianprops.linewidth'] = 2
     # Tick direction and padding
    mpl.rcParams['xtick.major.pad'] = 10
    mpl.rcParams['ytick.major.pad'] = 10
    # Spine line widths
    mpl.rcParams['axes.linewidth'] = 2  # This affects bottom and left spines
    # Tick line widths
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['boxplot.flierprops.linewidth'] = 1.5


def pandas_styleset():
    # configure pandas table display
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)


def save_figure(
    fig: Figure, 
    fig_dir: Union[Path, str], 
    file_name: str, 
    dpi: int = 300
) -> None:
    """
    Save figure to file
    Args:
        fig: Matplotlib figure to save
        fig_dir: Directory path to save the figure
        file_name: Name of the file to save
        dpi: Resolution in dots per inch
    """
    fig_dir = Path(fig_dir)  # Convert to Path if string
    fig_dir.mkdir(parents=True, exist_ok=True)
    filepath = fig_dir / file_name
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)



def plot_distribution(
    df: pd.DataFrame, 
    column: str, 
    xlabel: str, 
    fig_dir: Union[Path, str], 
    file_name: str, 
    bins: int = 30
) -> None:
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=bins)
    ax.set_xlabel(xlabel)
    save_figure(fig, fig_dir, file_name)
