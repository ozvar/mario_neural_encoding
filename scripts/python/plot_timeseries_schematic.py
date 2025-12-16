import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from mario_encoding.config import PATHS
from mario_encoding.utils import save_figure

def generate_timeseries_schematic(n_rows=4, n_timepoints=1000, colors=None, 
                                   output_path='timeseries_schematic.png',
                                   figsize=(10, 7), dpi=150):
    """
    Generate dummy time series for methods schematic.
    
    Parameters
    ----------
    n_rows : int
        Number of time series to generate
    n_timepoints : int
        Length of each time series
    colors : list of str, optional
        Color per row. If None, uses grey for all rows
    output_path : str
        Path to save figure
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution for saved figure
    """
    
    # Generate band-limited noise for each row
    timeseries = []
    for _ in range(n_rows):
        # White noise
        noise = np.random.randn(n_timepoints)
        # Low-pass filter to get smooth-ish fluctuations
        b, a = signal.butter(3, 0.1, btype='low')
        filtered = signal.filtfilt(b, a, noise)
        timeseries.append(noise)
    
    # Set colors
    if colors is None:
        colors = ['grey'] * n_rows
    elif len(colors) != n_rows:
        raise ValueError(f"colors length ({len(colors)}) != n_rows ({n_rows})")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (ts, color) in enumerate(zip(timeseries, colors)):
        # Offset each row vertically
        offset = i * 7
        ax.plot(ts + offset, color=color, linewidth=1.5, alpha=0.95)
    
    # Clean up axes
    ax.set_xlim(0, n_timepoints)
    ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, output_path, 'time_series_schematic')
    plt.close()
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    # Define color groups
    color1 = '#ec8e48'  # blue
    color2 = '#188bb7'  # orange
    color3 = '#9924bf'  # purple
    color4 = '#cd1616'  # red
    color5 = '#5fe23a'  # green
    colors = [color3] * 1 + [color4] * 1 + [color5] * 1
    output_path = PATHS['figure_assets']

    generate_timeseries_schematic(n_rows=3, colors=colors, output_path=output_path)
