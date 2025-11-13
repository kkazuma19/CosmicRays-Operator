import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cmocean
from s_deeponet import SequentialDeepONet, SequentialDeepONet_Dropout
from skimage.metrics import structural_similarity as ssim

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


def simulate_sensor_failure(input_seq, replace_value, sensor_indices):
    """
    Replace specified sensor channels with a scalar or per-sensor value.
    Works with numpy arrays or torch.Tensors. Returns same type as input.
    """
    import numpy as np
    import torch

    is_tensor = torch.is_tensor(input_seq)

    if is_tensor:
        out = input_seq.clone()
    else:
        out = input_seq.copy()

    # helper to get value for a given sensor index
    def _val_for(idx):
        if np.isscalar(replace_value):
            return replace_value
        try:
            return replace_value[idx]
        except Exception:
            return replace_value  # fallback

    for si in sensor_indices:
        v = _val_for(si)
        if is_tensor:
            # ensure v is a tensor on the same device/dtype
            if not torch.is_tensor(v):
                v = torch.as_tensor(v, dtype=out.dtype, device=out.device)
            out[:, :, si] = v
        else:
            out[:, :, si] = v

    return out



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

def init_model_dropout():
    ''' Initialize the model architecture '''
    dim = 128
    model = SequentialDeepONet_Dropout(
        branch_type='lstm',
        branch_input_size=12,
        branch_hidden_size=128,
        branch_num_layers=4,
        branch_output_size=dim,
        trunk_architecture=[2, 128, 128, dim],
        num_outputs=1,
        activation_fn=nn.ReLU,
        dropout=0.2
    )
    return model

def init_model_deeponet():
    ''' Initialize the model architecture '''
    dim = 128
    model = SequentialDeepONet(
        branch_type='fcn',
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

def load_model_experiment_dropout(model_path):
    ''' Load model from a given path '''
    model = init_model_dropout()
    model.load_state_dict(torch.load(model_path))
    
    # check if correctly loaded
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    model.eval()
    return model

def load_model_experiment_deeponet(model_path):
    ''' Load model from a given path '''
    model = init_model_deeponet()
    model.load_state_dict(torch.load(model_path))
    
    # check if correctly loaded
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    model.eval()
    return model


def fit(model, data_loader, device, scaler_target):
    all_outputs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in data_loader:
            branch_batch, trunk_batch, target_batch = (
                branch_batch.to(device),
                trunk_batch.to(device),
                target_batch.to(device),
            )
            output = model(branch_batch, trunk_batch)
            
            all_outputs.append(output.cpu())
            all_targets.append(target_batch.cpu())

    # ...existing code...
    # After loop:
    outputs = torch.cat(all_outputs, dim=0)  # [N_test, ...]
    targets = torch.cat(all_targets, dim=0)  # [N_test, ...]

    # move to numpy
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    # flatten to 2D (n_samples, n_features) for scaler
    out_shape = outputs.shape
    tgt_shape = targets.shape
    outputs_flat = outputs.reshape(outputs.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # inverse scale
    outputs_flat = scaler_target.inverse_transform(outputs_flat)
    targets_flat = scaler_target.inverse_transform(targets_flat)

    # restore original shapes
    outputs = outputs_flat.reshape(out_shape)
    targets = targets_flat.reshape(tgt_shape)

    
    return outputs, targets


def compute_metrics_region(pred_img, targ_img, lon_grid, lat_grid, region_extent):
    """
    Compute rel-L2 and SSIM restricted to a spatial subset.
    region_extent = [lon_min, lon_max, lat_min, lat_max]
    pred_img, targ_img: arrays of shape (N, H, W)
    lon_grid, lat_grid: (H, W)
    """
    lon_min, lon_max, lat_min, lat_max = region_extent

    # Mask to select region
    mask_lat = (lat_grid[:, 0] >= lat_min) & (lat_grid[:, 0] <= lat_max)
    mask_lon = (lon_grid[0, :] >= lon_min) & (lon_grid[0, :] <= lon_max)

    # Subset fields
    pred_sub = pred_img[:, mask_lat, :][:, :, mask_lon]
    targ_sub = targ_img[:, mask_lat, :][:, :, mask_lon]

    # Flatten spatial dims
    N = pred_sub.shape[0]
    eps = 1e-12
    pf = pred_sub.reshape(N, -1)
    tf = targ_sub.reshape(N, -1)

    # --- Relative L2 (%) ---
    rel_l2_pct = 100.0 * np.linalg.norm(pf - tf, axis=1) / (np.linalg.norm(tf, axis=1) + eps)

    # --- SSIM ---
    ssim_scores = np.empty(N, dtype=float)
    for i in range(N):
        dr = float(targ_sub[i].max() - targ_sub[i].min()) or 1.0
        ssim_scores[i] = ssim(targ_sub[i], pred_sub[i], data_range=dr)

    print(f"Region {region_extent}")
    print("Mean rel-L2 (%):", rel_l2_pct.mean())
    print("Mean SSIM:", ssim_scores.mean())
    
    return rel_l2_pct, ssim_scores


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

# Atlantic-centered box (0°)
# import cmocean
# cmap_seq = cmocean.cm.thermal
# plot_global_field_box(
#     lon_grid, lat_grid, pred_img, i=10,
#     title="Prediction (orig units)",
#     units_label="Effective Dose Rate (μSv/h)",
#     cmap=cmap_seq, central_longitude=0,
#     savepath="analysis/zero-shot/pred_box_0.png"
# )


def load_dose_years(data_dir: str,
                    start_year: int = 2001,
                    end_year: int = 2023,
                    pattern: str = "dose_{year}_0m.npy",
                    concat_axis: int = 0,
                    allow_missing: bool = True,
                    verbose: bool = True):
    """
    Load yearly dose .npy files and concatenate them.

    Parameters
    ----------
    data_dir : str
        Directory containing files.
    start_year, end_year : int
        Inclusive year range to load.
    pattern : str
        Filename pattern with '{year}' placeholder.
    concat_axis : int
        Axis along which to concatenate (default 0 -> time axis).
    allow_missing : bool
        If True, skip missing files and continue; if False, raise FileNotFoundError.
    verbose : bool
        Print progress.

    Returns
    -------
    combined : np.ndarray
        Concatenated array (time, lat, lon).
    years_loaded : list[int]
        Years that were successfully loaded (in order).
    """
    import os
    import numpy as np

    arrays = []
    years_loaded = []
    for y in range(start_year, end_year + 1):
        fname = os.path.join(data_dir, pattern.format(year=y))
        if not os.path.exists(fname):
            if allow_missing:
                if verbose:
                    print(f"Warning: file missing, skipping: {fname}")
                continue
            else:
                raise FileNotFoundError(f"Missing required file: {fname}")
        if verbose:
            print(f"Loading {fname} ...")
        a = np.load(fname)
        arrays.append(a)
        years_loaded.append(y)

    if len(arrays) == 0:
        raise RuntimeError("No files loaded. Check data_dir and pattern.")

    # basic sanity: ensure shapes match except concat_axis
    ref_shape = list(arrays[0].shape)
    for idx, a in enumerate(arrays[1:], start=1):
        shp = list(a.shape)
        # allow different size on concat_axis; require others equal
        for ax, (r, s) in enumerate(zip(ref_shape, shp)):
            if ax == concat_axis:
                continue
            if r != s:
                raise ValueError(f"Incompatible shape for year {years_loaded[idx]}: {shp} vs {ref_shape}")

    combined = np.concatenate(arrays, axis=concat_axis)
    if verbose:
        print(f"Combined shape: {combined.shape} (years: {years_loaded})")
    return combined, years_loaded


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_field_region(
    lon_grid, lat_grid, field, i=0, *,
    title=None, units_label="value",
    cmap="viridis", vmin=None, vmax=None,
    region_extent=None,     # [lon_min, lon_max, lat_min, lat_max]
    show_coast=True, show_borders=True,
    tick_step=(10, 5),
    add_contour=False, contour_levels=8, contour_color="k", contour_alpha=0.8,
    mark_equator_meridian=True,
    savepath=None, dpi=300, close=True
):
    """Regional map with *exactly aligned* colorbar under the plot."""
    assert field.ndim == 3 and 0 <= i < field.shape[0]
    lon_min, lon_max, lat_min, lat_max = region_extent

    # --- Subset region ---
    mask_lat = (lat_grid[:, 0] >= lat_min) & (lat_grid[:, 0] <= lat_max)
    mask_lon = (lon_grid[0, :] >= lon_min) & (lon_grid[0, :] <= lon_max)
    sub_field = field[i][np.ix_(mask_lat, mask_lon)]
    sub_lon = lon_grid[np.ix_(mask_lat, mask_lon)]
    sub_lat = lat_grid[np.ix_(mask_lat, mask_lon)]

    # --- Setup ---
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': proj})

    # --- Base map ---
    if show_coast:
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.coastlines(resolution="50m", lw=0.6)
    if show_borders:
        ax.add_feature(cfeature.BORDERS, lw=0.4)

    # --- Main field ---
    im = ax.pcolormesh(
        sub_lon, sub_lat, sub_field,
        transform=proj, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )

    # --- Contours ---
    if add_contour:
        cs = ax.contour(
            sub_lon, sub_lat, sub_field,
            levels=contour_levels, colors=contour_color,
            linewidths=0.4, alpha=contour_alpha, transform=proj
        )
        ax.clabel(cs, fmt="%.3f", fontsize=6, inline=True, inline_spacing=2)

    # --- Set extent ---
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # --- Manual ticks ---
    lon_step, lat_step = tick_step
    xticks = np.arange(lon_min, lon_max + lon_step, lon_step)
    yticks = np.arange(lat_min, lat_max + lat_step, lat_step)
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)

    def fmt_lon(x):
        return f"{abs(int(x))}°{'W' if x < 0 else 'E' if x > 0 else ''}"

    def fmt_lat(y):
        return f"{abs(int(y))}°{'S' if y < 0 else 'N' if y > 0 else ''}"

    ax.set_xticklabels([fmt_lon(x) for x in xticks], fontsize=9)
    ax.set_yticklabels([fmt_lat(y) for y in yticks], fontsize=9)
    ax.tick_params(length=3, direction="out")

    # --- Optional equator/meridian lines ---
    if mark_equator_meridian:
        ax.plot([lon_min, lon_max], [0, 0], transform=proj, color="gray", lw=0.5, ls="--")
        if lon_min < 0 < lon_max:
            ax.plot([0, 0], [lat_min, lat_max], transform=proj, color="gray", lw=0.5, ls="--")

    # --- Title ---
    if title:
        ax.set_title(title, fontsize=12, pad=6)

    # ======================================================
    #  COLORBAR PERFECTLY ALIGNED AFTER DRAW
    # ======================================================
    plt.tight_layout()
    fig.canvas.draw_idle()  # ensure layout is applied
    renderer = fig.canvas.get_renderer()
    pos = ax.get_position()  # [x0, y0, width, height] in figure coords

    # exact alignment with map
    cbar_height = 0.04
    cbar_bottom = pos.y0 - 0.15  # spacing below map
    cbar_left = pos.x0
    cbar_width = pos.width

    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(units_label, fontsize=10)
    cb.ax.tick_params(labelsize=8)

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    if close:
        plt.close(fig)

import datetime  # <--- NEW import

def plot_three_samples_pred_err(
    lon_grid, lat_grid,
    pred, target, idx_list,
    *,
    region_extent=None,              
    units_pred="Effective Dose Rate (µSv/h)",
    units_err="Error (µSv/h)",
    cmap_pred="viridis",
    cmap_err="coolwarm",
    tick_step=(30, 15),
    show_coast=True,
    show_borders=True,
    mark_equator_meridian=True,
    error_mode="diff",               
    err_symmetry_pad=0.02,           
    vmin_pred=None, vmax_pred=None,
    vlim_err=None,                   
    add_contour=False,
    contour_levels=8,
    contour_color="k",
    contour_alpha=0.8,
    start_date="2023-01-01",  # <--- NEW: base date for index conversion
    dpi=300, savepath=None, close=True
):
    """
    2x3 figure: columns = indices in idx_list.
    Top row: predictions (with contour + labels).
    Bottom row: error maps (with contour, no labels).
    Titles show calendar day based on start_date and index.
    """
    assert pred.ndim == 3 and target.ndim == 3
    assert len(idx_list) == 3
    assert region_extent is not None

    lon_min, lon_max, lat_min, lat_max = region_extent

    # --- Prepare date converter ---
    base_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    # --- Subset region ---
    mask_lat = (lat_grid[:, 0] >= lat_min) & (lat_grid[:, 0] <= lat_max)
    mask_lon = (lon_grid[0, :] >= lon_min) & (lon_grid[0, :] <= lon_max)
    sub_lon = lon_grid[np.ix_(mask_lat, mask_lon)]
    sub_lat = lat_grid[np.ix_(mask_lat, mask_lon)]

    # --- Extract and compute errors ---
    P = [pred[i][np.ix_(mask_lat, mask_lon)] for i in idx_list]
    T = [target[i][np.ix_(mask_lat, mask_lon)] for i in idx_list]
    E = [p - t if error_mode == "diff" else np.abs(p - t) for p, t in zip(P, T)]

    # --- Color limits ---
    if vmin_pred is None or vmax_pred is None:
        allP = np.stack(P)
        vmin_pred = np.nanmin(allP) * 0.9 if vmin_pred is None else vmin_pred
        vmax_pred = np.nanmax(allP) * 1.1 if vmax_pred is None else vmax_pred

    if error_mode == "diff":
        if vlim_err is None:
            allE = np.stack(E)
            amax = np.nanmax(np.abs(allE)) * (1.0 + err_symmetry_pad)
        else:
            amax = vlim_err
        vmin_err, vmax_err = -amax, +amax
    else:
        allE = np.stack(E)
        vmin_err, vmax_err = 0.0, np.nanmax(allE)

    # --- Figure setup ---
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=dpi,
                            subplot_kw={'projection': proj}, constrained_layout=False)

    def add_basemap(ax):
        if show_coast:
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
            ax.coastlines(resolution="50m", lw=0.6)
        if show_borders:
            ax.add_feature(cfeature.BORDERS, lw=0.4)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        lon_step, lat_step = tick_step
        xticks = np.arange(lon_min, lon_max + 1e-9, lon_step)
        yticks = np.arange(lat_min, lat_max + 1e-9, lat_step)
        ax.set_xticks(xticks, crs=proj)
        ax.set_yticks(yticks, crs=proj)
        def fmt_lon(x): return f"{abs(int(x))}°{'W' if x < 0 else 'E' if x > 0 else ''}"
        def fmt_lat(y): return f"{abs(int(y))}°{'S' if y < 0 else 'N' if y > 0 else ''}"
        ax.set_xticklabels([fmt_lon(x) for x in xticks], fontsize=9)
        ax.set_yticklabels([fmt_lat(y) for y in yticks], fontsize=9)
        ax.tick_params(length=3, direction="out")
        if mark_equator_meridian:
            ax.plot([lon_min, lon_max], [0, 0], transform=proj, color="gray", lw=0.5, ls="--")
            if lon_min < 0 < lon_max:
                ax.plot([0, 0], [lat_min, lat_max], transform=proj, color="gray", lw=0.5, ls="--")

    # --- Draw panels ---
    ims_pred, ims_err = [], []
    for col, idx in enumerate(idx_list):
        # Compute calendar date
        current_date = base_date + datetime.timedelta(days=int(idx))
        date_label = current_date.strftime("2023-%m-%d")

        # Optional percentile annotation
        percentile_labels = ["5th-percentile", "50th-percentile", "95th-percentile"]
        percentile_text = percentile_labels[col] if col < len(percentile_labels) else ""

        # --- Top: Prediction ---
        axP = axs[0, col]
        add_basemap(axP)
        imP = axP.pcolormesh(sub_lon, sub_lat, P[col], transform=proj,
                             cmap=cmap_pred, vmin=vmin_pred, vmax=vmax_pred, shading="auto")
        if add_contour:
            cs = axP.contour(sub_lon, sub_lat, P[col], levels=contour_levels,
                             colors=contour_color, linewidths=0.4, alpha=contour_alpha, transform=proj)
            axP.clabel(cs, fmt="%.3f", fontsize=6, inline=True, inline_spacing=2)

        # UPDATED TITLE LINE
        #axP.set_title(f"Prediction ({date_label}, {percentile_text})", fontsize=11, pad=6)
        axP.set_title(f"Prediction ({percentile_text}, {date_label})", fontsize=11, pad=6)
        ims_pred.append(imP)

        # --- Bottom: Error ---
        axE = axs[1, col]
        add_basemap(axE)
        imE = axE.pcolormesh(sub_lon, sub_lat, E[col], transform=proj,
                             cmap=cmap_err, vmin=vmin_err, vmax=vmax_err, shading="auto")
        if add_contour:
            axE.contour(sub_lon, sub_lat, E[col], levels=contour_levels,
                        colors=contour_color, linewidths=0.4, alpha=contour_alpha, transform=proj)
        #axE.set_title("Error Map", fontsize=11, pad=6)
        ims_err.append(imE)


    # --- Shared colorbars ---
    plt.tight_layout()
    fig.canvas.draw()

    def add_row_cbar(row_axes, im, label, y_gap=0.06, height=0.03):
        left = min(ax.get_position().x0 for ax in row_axes)
        right = max(ax.get_position().x1 for ax in row_axes)
        bottom = min(ax.get_position().y0 for ax in row_axes)
        cax = fig.add_axes([left, bottom - y_gap, right - left, height])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=14)
        cb.ax.tick_params(labelsize=8)
        return cb

    add_row_cbar(axs[0, :], ims_pred[0], units_pred, y_gap=0.070, height=0.028)
    add_row_cbar(axs[1, :], ims_err[0], units_err, y_gap=0.070, height=0.028)

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    if close:
        plt.close(fig)