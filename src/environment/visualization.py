import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from typing import List, Optional
from geometry import DesignRegion, Receiver

_RECEIVER_COLOR = 'purple'


def visualize_wave_propagation(e_z: np.ndarray, canvas: np.ndarray,
                               receivers: Optional[List[Receiver]] = None):
    """
    Visualizes the real part of the E_z field as a red-blue diverging plot,
    showing the wave propagation pattern in space.

    Coordinate System:
    --> Input arrays use numpy ndarray convention (row, col). The function
        transposes them internally to match Cartesian display where x is
        horizontal and y is vertical (going upward).

    If canvas is provided, draws contour lines around regions of different
    permittivity to overlay rod positions on top of the wave plot.
    If receivers are provided, draws their outlines as colored contours
    using each receiver's _mask (populated by initialize_environment).

    Inputs:
    --> e_z [np.ndarray]: Complex-valued 2D E_z field array
    --> canvas [np.ndarray]: Permittivity array for overlaying rod outlines
    --> receivers [Optional[List[Receiver]]]: Receivers to outline

    Returns:
    --> None: Displays the plot via plt.show()
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    # Get the real component of the E_z field for the wave visualization
    real_field = np.real(e_z) # [unitless]

    # Center the colormap at zero using 98th percentile to avoid extreme outliers
    # This makes colors more vivid for the bulk of the data
    vmax = np.percentile(np.abs(e_z), 98) # [unitless]

    # Plot the real field as a diverging red-blue colormap
    image = ax.imshow(real_field, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)

    # Overlay rod and wall outlines at appropriate permittivity boundaries
    # Rods (permittivity ~5.0) shown at contour between background (1.0) and rods
    # Walls (permittivity ~1e6) shown at contour between rods and walls
    contour_level_rods = 3.0  # midpoint between background (1.0) and rods (5.0)
    contour_level_walls = 5e5  # midpoint between rods (5.0) and walls (1e6)
    ax.contour(canvas, [contour_level_rods, contour_level_walls], colors='k', alpha=0.5)

    # Overlay receiver outlines
    if receivers:
        for receiver in receivers:
            ax.contour(receiver._mask, [0.5], colors=_RECEIVER_COLOR,
                       alpha=0.9, linewidths=1.2)
        ax.legend(
            handles=[Line2D([0], [0], color=_RECEIVER_COLOR, lw=1.5, label='Receiver')],
            loc='upper right', fontsize=8, framealpha=0.85,
        )

    # Label the axes for clarity
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Wave Propagation (Re(E_z))')

    # Add a color bar to interpret field values
    plt.colorbar(image, ax=ax)

    # Display the plot
    plt.show()


def visualize_ez_intensity(e_z: np.ndarray, canvas: np.ndarray = None,
                           receivers: Optional[List[Receiver]] = None):
    """
    Visualizes the intensity (magnitude) of the E_z field as a heatmap.

    Intensity is computed as |E_z|^2 (squared magnitude, showing energy density).
    If canvas is provided, overlays the permittivity structure outlines.
    If receivers are provided, draws their outlines as colored contours.

    Inputs:
    --> e_z [np.ndarray]: Complex-valued 2D E_z field array
    --> canvas [np.ndarray]: Optional permittivity array for overlaying structure
    --> receivers [Optional[List[Receiver]]]: Receivers to outline

    Returns:
    --> None: Displays the plot via plt.show()
    """
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    # Compute field intensity as |E_z|^2 (energy density)
    intensity = np.abs(e_z) ** 2

    # Plot intensity as heatmap using 98th percentile for vibrant colors
    vmax_intensity = np.percentile(intensity, 98)
    image = ax.imshow(intensity, cmap='inferno', origin='lower', vmin=0, vmax=vmax_intensity)

    # Overlay rod and wall outlines if canvas is provided
    if canvas is not None:
        contour_level_rods = 3.0
        contour_level_walls = 5e5
        ax.contour(canvas, [contour_level_rods, contour_level_walls], colors='white', alpha=0.6, linewidths=0.5)

    # Overlay receiver outlines
    if receivers:
        for receiver in receivers:
            ax.contour(receiver._mask, [0.5], colors='white',
                       alpha=0.9, linewidths=1.2)
        ax.legend(
            handles=[Line2D([0], [0], color='white', lw=1.5, label='Receiver')],
            loc='upper right', fontsize=8, framealpha=0.85,
        )

    # Label axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Field Intensity (|E_z|²)')

    # Add colorbar
    plt.colorbar(image, ax=ax, label='Intensity (V²/m²)')

    # Display
    plt.show()


def visualize_permittivity_map(canvas: np.ndarray):
    """
    Visualizes the permittivity structure (material distribution) in the design region.

    Shows background material, rods, and walls with a material-focused colormap.

    Inputs:
    --> canvas [np.ndarray]: Permittivity array

    Returns:
    --> None: Displays the plot via plt.show()
    """
    _, ax = plt.subplots(1, 1, constrained_layout=True)

    # Clip permittivity to [0,10] so the huge wall permittivity (1e6) doesn't
    # dominate; vacuum sits near 0, rods at ~5, walls clip to the top.
    canvas_clipped = np.clip(canvas, 0, 10)

    # Blue scale: transparent/vacuum (low ε) -> black, dielectric rods -> blue,
    # metallic walls (high ε) -> white, on a black background.
    from matplotlib.colors import LinearSegmentedColormap
    blue_metallic = LinearSegmentedColormap.from_list(
        "blue_metallic", ["#000000", "#1f5fff", "#ffffff"])
    ax.set_facecolor("black")
    # Map physical permittivity [0,10] -> normalized [-1,1] for display so the
    # colorbar reads on the [-1,1] scale (0->-1, 5->0, 10->+1); appearance is
    # unchanged (vacuum navy, rods blue, walls white).
    image = ax.imshow(canvas_clipped / 5.0 - 1.0, cmap=blue_metallic,
                      origin='lower', vmin=-1.0, vmax=1.0)

    # Label axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Permittivity Distribution')

    # Add colorbar
    plt.colorbar(image, ax=ax, label='Permittivity (ε)')

    # Add text annotation explaining the colors
    ax.text(0.02, 0.98, 'Black: Vacuum (ε≈1.0)\nBlue: Rods (ε≈5.0)\nWhite: Walls (ε≥10, metallic)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)

    # Display
    plt.show()