import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cmocean
from s_deeponet import SequentialDeepONet

#print("Using helper from analysis/zero-shot/helper.py")

def train_val_test_split(input_data, target):
    # Define the number of test samples (last 365 days)
    test_size = 365

    # Split data into training+validation and test
    train_val_input = input_data[:-test_size]
    train_val_target = target[:-test_size]
    test_input = input_data[-test_size:]
    test_target = target[-test_size:]

    # Calculate split index for training and validation
    train_size = int(len(train_val_input) * 0.5)  
    val_size = len(train_val_input) - train_size  

    # Training set
    train_input = train_val_input[:train_size]
    train_target = train_val_target[:train_size]

    # Validation set
    val_input = train_val_input[train_size:]
    val_target = train_val_target[train_size:]

    # Final shapes check
    print("Train input shape:", train_input.shape)
    print("Validation input shape:", val_input.shape)
    print("Test input shape:", test_input.shape)

    return train_input, train_target, val_input, val_target, test_input, test_target


def init_model():
    ''' Initialize the model architecture '''
    dim = 128
    model = SequentialDeepONet(
        branch_type='lstm',
        branch_input_size=12,
        branch_hidden_size=128,
        branch_num_layers=4,
        branch_output_size=dim,
        trunk_architecture=[2, 128, 128, dim],
        num_outputs=1,
        use_transform=False,
        activation_fn=nn.ReLU,
    )
    return model


def load_model_experiment(model_path):
    ''' Load model from a given path '''
    model = init_model()
    model.load_state_dict(torch.load(model_path))
    
    # check if correctly loaded
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    model.eval()
    return model


def extract_region_from_npy(
    file_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    grid_step: float = 0.25,
):
    """
    Extract a latitude–longitude box from an EXPACS .npy dataset.

    Parameters
    ----------
    file_path : str
        Path to the .npy file (shape: [days, gridpoints]).
    lat_min, lat_max : float
        Latitude bounds in degrees.
    lon_min, lon_max : float
        Longitude bounds in degrees.
    grid_step : float, optional
        Grid resolution in degrees (default = 0.25).

    Returns
    -------
    region_dose : np.ndarray
        Dose array for the region, shape (days, n_lat, n_lon).
    region_lats : np.ndarray
        1D array of latitudes in the region.
    region_lons : np.ndarray
        1D array of longitudes in the region.
    """

    # Load full data
    dose = np.load(file_path, mmap_mode="r")  # shape: (days, gridpoints)
    print(f"Loaded {file_path}: {dose.shape[0]} days × {dose.shape[1]} gridpoints")

    # Build coordinate grid (must match your input generation)
    longitudes = np.arange(-180, 180 + grid_step, grid_step)
    latitudes  = np.arange(-90,  90 + grid_step, grid_step)
    n_lat, n_lon = len(latitudes), len(longitudes)

    # Reshape to 3D (days, lat, lon)
    dose_3d = dose.reshape(-1, n_lat, n_lon)

    # Masks for region
    lat_mask = (latitudes >= lat_min) & (latitudes <= lat_max)
    lon_mask = (longitudes >= lon_min) & (longitudes <= lon_max)

    # Extract region
    region_dose = dose_3d[:, lat_mask][:, :, lon_mask]
    region_lats = latitudes[lat_mask]
    region_lons = longitudes[lon_mask]

    print(f"Extracted region: {lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E "
          f"→ shape {region_dose.shape}")

    return region_dose, region_lats, region_lons

def convert2dim(dose_array, grid_path='data/grid_points.npy'):
    """
    dose_array: shape (N, P) or (P,) where P == n_lat * n_lon (same order as grid_points)
    grid_points.npy: shape (P, 2) with columns [lat, lon]
    
    Returns:
      lon_grid: (H, W)
      lat_grid: (H, W)
      Z:        (N, H, W)   # if input was (P,), N=1
    """
    grid_array = np.load(grid_path)              # (P, 2) [lat, lon]
    # sort by (lat, lon)
    sorted_idx = np.lexsort((grid_array[:, 1], grid_array[:, 0]))
    sorted_grid = grid_array[sorted_idx]

    lats = np.unique(sorted_grid[:, 0])
    lons = np.unique(sorted_grid[:, 1])
    H, W = len(lats), len(lons)

    # Make mesh
    lon_grid, lat_grid = np.meshgrid(lons, lats) # (H, W)

    # Ensure batch dimension
    dose = np.asarray(dose_array)
    if dose.ndim == 1:
        dose = dose[None, ...]                   # (1, P)

    # Reorder to (N, P) matching the sorted grid, then reshape to (N, H, W)
    dose_sorted = dose[:, sorted_idx]            # (N, P)
    Z = dose_sorted.reshape(dose.shape[0], H, W) # (N, H, W)
    return lon_grid, lat_grid, Z

# helper for plotting
def _edges_from_centers(centers):
    centers = np.asarray(centers)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be 1D with >=2 elements")
    edges = np.empty(centers.size + 1, dtype=centers.dtype)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

def plot_global_field_cartopy(
    lon_grid, lat_grid, field, i=0, *,
    title=None, units_label="value",
    cmap="viridis", vmin=None, vmax=None,
    projection="Robinson", coastline=True, borders=True,
    use_pcolormesh=True, gaussian_filter=False,
    savepath=None, dpi=300, close=True
):
    """
    Plot a single timestep from a (N,H,W) field on a global map with Cartopy.

    Args:
        lon_grid, lat_grid: (H,W) arrays of lon/lat centers (from convert2dim)
        field: (N,H,W) data array
        i: timestep index to plot
        title: plot title
        units_label: colorbar label
        cmap: matplotlib colormap name
        vmin, vmax: color limits
        projection: 'Robinson' or 'PlateCarree'
        coastline, borders: draw coast/borders
        use_pcolormesh: True (recommended) or False to use imshow
        gaussian_filter: (not applied; placeholder if you later smooth)
        savepath: if given, save to this path; else just show
        dpi: figure DPI for saving
        close: if True, closes the figure after save/show
    """
    # Validate shapes
    H, W = lon_grid.shape
    assert lat_grid.shape == (H, W), "lat_grid shape mismatch"
    assert field.ndim == 3 and field.shape[1:] == (H, W), "field must be (N,H,W)"
    assert 0 <= i < field.shape[0], "timestep index out of range"

    # 1D lon/lat centers from the 2D grids
    lon_centers = lon_grid[0, :]
    lat_centers = lat_grid[:, 0]

    # Select the map projection
    proj_map = ccrs.Robinson() if projection.lower() == "robinson" else ccrs.PlateCarree()
    proj_data = ccrs.PlateCarree()

    # Data slice
    Z = np.asarray(field[i], dtype=float)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj_map)
    ax.set_global()

    if coastline:
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.coastlines(lw=0.7)
    if borders:
        ax.add_feature(cfeature.BORDERS, lw=0.4)

    # Gridlines (no labels for cleaner look; toggle if needed)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="k", alpha=0.2, linestyle="--")

    if use_pcolormesh:
        # Safer geospatial path: build edges, add a cyclic column to avoid the dateline seam
        Z_cyc, lon_cyc = add_cyclic_point(Z, coord=lon_centers)
        lon_edges = _edges_from_centers(lon_cyc)
        lat_edges = _edges_from_centers(lat_centers)
        Lon, Lat = np.meshgrid(lon_edges, lat_edges)
        im = ax.pcolormesh(Lon, Lat, Z_cyc, transform=proj_data,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    else:
        # Fast path with imshow (assumes regular spacing); set extent explicitly
        Z_cyc, lon_cyc = add_cyclic_point(Z, coord=lon_centers)
        lon_min, lon_max = float(lon_cyc.min()), float(lon_cyc.max())
        lat_min, lat_max = float(lat_centers.min()), float(lat_centers.max())
        im = ax.imshow(Z_cyc, origin="lower",
                       extent=[lon_min, lon_max, lat_min, lat_max],
                       transform=proj_data, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
    cb.set_label(units_label)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    if close:
        plt.close(fig)

def plot_global_field_box(
    lon_grid, lat_grid, field, i=0, *,
    title=None, units_label="value",
    cmap="viridis", vmin=None, vmax=None,
    central_longitude=0,
    show_coast=True, show_borders=True,
    show_ticks=True, tick_step=(60, 30),
    use_pcolormesh=True,
    savepath=None, dpi=300, close=True
):
    """
    Plate Carrée 'box' map plotter for a single timestep from (N,H,W).

    lon_grid, lat_grid: (H,W)
    field: (N,H,W)
    i: timestep index
    """
    H, W = lon_grid.shape
    assert field.ndim == 3 and field.shape[1:] == (H, W)
    assert 0 <= i < field.shape[0]

    lon_centers = lon_grid[0, :]
    lat_centers = lat_grid[:, 0]

    proj_map  = ccrs.PlateCarree(central_longitude=central_longitude)
    proj_data = ccrs.PlateCarree()

    Z = np.asarray(field[i], dtype=float)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj_map)

    # Base map
    if show_coast:
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.coastlines(lw=0.7)
    if show_borders:
        ax.add_feature(cfeature.BORDERS, lw=0.4)

    # Plot data (handle wrap seam)
    if use_pcolormesh:
        Z_cyc, lon_cyc = add_cyclic_point(Z, coord=lon_centers)
        lon_edges = _edges_from_centers(lon_cyc)
        lat_edges = _edges_from_centers(lat_centers)
        Lon, Lat = np.meshgrid(lon_edges, lat_edges)
        im = ax.pcolormesh(Lon, Lat, Z_cyc, transform=proj_data,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_extent([lon_centers.min(), lon_centers.max(),
                       lat_centers.min(), lat_centers.max()],
                      crs=proj_data)
    else:
        Z_cyc, lon_cyc = add_cyclic_point(Z, coord=lon_centers)
        im = ax.imshow(Z_cyc, origin="lower",
                       extent=[lon_cyc.min(), lon_cyc.max(),
                               lat_centers.min(), lat_centers.max()],
                       transform=proj_data,
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    # Gridlines and ticks
    if show_ticks:
        lon_step, lat_step = tick_step
        xticks = np.arange(-180, 181, lon_step)
        yticks = np.arange(-90,   91,  lat_step)
        ax.set_xticks(xticks, crs=proj_data)
        ax.set_yticks(yticks, crs=proj_data)
        lon_formatter = LongitudeFormatter(number_format='.0f')
        lat_formatter = LatitudeFormatter(number_format='.0f')
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.gridlines(draw_labels=False, linewidth=0.4, color="k",
                     alpha=0.2, linestyle="--")

    if title:
        ax.set_title(title, pad=8)

    # ---- AUTO-WIDTH COLORBAR ----
    fig.canvas.draw()
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x0,                # align with left edge of map
        pos.y0 - 0.075,         # below map
        pos.width,             # match width dynamically
        0.03                   # fixed height
    ])
    cb = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cb.set_label(units_label)

    # Adjust layout safely (no tight_layout warnings)
    fig.subplots_adjust(bottom=0.12, top=0.92)

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    if close:
        plt.close(fig)
