#!/usr/bin/env python3
# ΔU and ΔV (SC − CC) contour maps for Span_1‥24
# x∈[120,155] mm, y∈[0,4] mm ;  X-axis nondimensionalised with convert_sx
# Colour scale fixed: −8 (blue) … 0 (white) … +3.5 (red)
# Wide-and-thin figure: 10 in × 2.5 in
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as patches
from dimensionalization import convert_sx

# ── constants ───────────────────────────────────────────────────────────────
X_MIN, X_MAX = 120, 155           # mm
Y_MIN, Y_MAX =   0,   4           # mm
LEVELS       = np.linspace(-0.2, 0.2, 257)   # 256 uniform bands
NORM         = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
OUT_DIR      = "images_diff"

# ── helpers ────────────────────────────────────────────────────────────────
def _read_as_float(csv_path):
    """Read CSV -> DataFrame with float axes, or None if file missing/empty."""
    if not os.path.isfile(csv_path):
        print(f"missing file: {csv_path}")
        return None
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        print(f"empty file: {csv_path}")
        return None
    df.index   = df.index.astype(float)    # Y
    df.columns = df.columns.astype(float)  # X
    return df

def _window(df):
    """Slice DataFrame to analysis rectangle."""
    return df.loc[
        (df.index   >= Y_MIN) & (df.index   <= Y_MAX),
        (df.columns >= X_MIN) & (df.columns <= X_MAX)
    ]

def _plot(title, file_png, Xdim, Y, diff):
    """Wide, thin contour plot with four strips and fixed −8…+3.5 scale."""
    plt.figure(figsize=(10, 2))

    cf = plt.contourf(
        Xdim, Y, diff,
        levels=np.linspace(-0.2, 0.2, 257),
        cmap='bwr',
        norm=TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2),
        extend='both'
    )
    # colour-bar with only min / max ticks
    cbar = plt.colorbar(cf)
    cbar.set_ticks([-0.2, 0.2])
    cbar.set_ticklabels(['-0.2', '0.2'])

    ax = plt.gca()

    # --- compute desired limits BEFORE adding strips ---------------------------
    xlim_target = (Xdim.min(), Xdim.max())     # data range only

    # ----- four black strips ---------------------------------------------------
    width   = 1.4 / 900
    spacing = 9.2 / 900
    height  = 0.17
    x_left0 = 0.125
    centre0 = x_left0 + width/2

    for k in range(4):
        centre = centre0 + k*spacing
        left   = centre - width/2
        ax.add_patch(
            patches.Rectangle(
                (left, Y.min()),      # bottom-left
                width,
                height,
                color='black',
                clip_on=True          # still good practice
            )
        )

    # --- restore original X-limits so patch is cropped -------------------------
    ax.set_xlim(xlim_target)            # <<–– keeps rectangle but crops excess



    plt.xlabel("Dimensionless X (x/c)")
    plt.ylabel("Y [mm]")
    plt.title(title)
    plt.savefig(file_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_png}")
# ---------------------------------------------------------------------------


# ── single span ────────────────────────────────────────────────────────────
def process_span(i_span: int):
    cc_u = f"PIV_planes/Case_CC_Span_{i_span}.txt_u.csv"
    sc_u = f"PIV_planes/Case_SC_Span_{i_span}.txt_u.csv"
    cc_v = f"PIV_planes/Case_CC_Span_{i_span}.txt_v.csv"
    sc_v = f"PIV_planes/Case_SC_Span_{i_span}.txt_v.csv"

    # Read all four; skip span if any missing
    dfs = {name: _read_as_float(path) for name, path in
           [('cc_u', cc_u), ('sc_u', sc_u), ('cc_v', cc_v), ('sc_v', sc_v)]}
    if any(df is None for df in dfs.values()):
        print(f"Span {i_span:02d}: skipped (missing file)")
        return

    # U-component
    diff_u = _window(dfs['sc_u']).values - _window(dfs['cc_u']).values
    Xdim_u, Y_u = np.meshgrid(
        convert_sx(_window(dfs['cc_u']).columns.values),
        _window(dfs['cc_u']).index.values
    )
    _plot(
        title   = f"Span {i_span:02d}  ΔU (SC – CC)",
        file_png= os.path.join(OUT_DIR, f"Span_{i_span:02d}_U.png"),
        Xdim    = Xdim_u,
        Y       = Y_u,
        diff    = diff_u
    )

    # V-component
    diff_v = _window(dfs['sc_v']).values - _window(dfs['cc_v']).values
    Xdim_v, Y_v = np.meshgrid(
        convert_sx(_window(dfs['cc_v']).columns.values),
        _window(dfs['cc_v']).index.values
    )
    _plot(
        title   = f"Span {i_span:02d}  ΔV (SC – CC)",
        file_png= os.path.join(OUT_DIR, f"Span_{i_span:02d}_V.png"),
        Xdim    = Xdim_v,
        Y       = Y_v,
        diff    = diff_v
    )

# ── batch driver ───────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for span in range(1, 25):
        process_span(span)

if __name__ == "__main__":
    main()