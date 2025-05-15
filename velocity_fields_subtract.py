import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from dimensionalization import convert_sx

# ------------------------ constants ----------------------------------------
X_MIN, X_MAX = 120, 155          # mm
Y_MIN, Y_MAX =   0,   4          # mm
LEVELS       = np.linspace(-0.2, 0.2, 257)   # 256 equal bands
NORM         = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
FIGSIZE      = (9, 3)         # wide & thin
OUT_DIR      = "images_diff"

# ------------------------ helpers ------------------------------------------
def _read_as_float(csv_path):
    """Read CSV; return DataFrame with float axes or None if missing/empty."""
    if not os.path.isfile(csv_path):
        print(f"missing file: {csv_path}")
        return None
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        print(f"empty file: {csv_path}")
        return None
    df.index   = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df

def _window(df):
    return df.loc[
        (df.index   >= Y_MIN) & (df.index   <= Y_MAX),
        (df.columns >= X_MIN) & (df.columns <= X_MAX)
    ]

def _plot(Xdim, Y, diff, title, out_png):
    """Wide‑thin contour with 4 strips, fixed scale, caption on colour‑bar."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    cf = ax.contourf(
        Xdim, Y, diff,
        levels=LEVELS,
        cmap='bwr',
        norm=NORM,
        extend='both'
    )

    # colour‑bar with only min/max ticks and caption
    cbar = fig.colorbar(cf)
    cbar.set_ticks([-0.2, 0.2])
    cbar.set_ticklabels(['-0.2', '0.2'])
    cbar.set_label("Velocity difference (SC – CC) [m/s]", fontsize=9)

    # ---- four black strips -------------------------------------------------
    width   = 1.4 / 900           # dimensionless x/c
    spacing = 9.2 / 900
    height  = 0.17                # mm
    x_left0 = 0.125
    centre0 = x_left0 + width / 2

    for k in range(4):
        left = centre0 + k*spacing - width/2
        ax.add_patch(
            patches.Rectangle(
                (left, Y.min()),  # bottom‑left
                width, height,
                color='black',
                clip_on=True      # excess cropped
            )
        )

    ax.set_xlim(Xdim.min(), Xdim.max())
    ax.set_xlabel("Dimensionless X (x/c)")
    ax.set_ylabel("Y [mm]")
    #ax.set_title(title)

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")

# -------------------- single span ------------------------------------------
def process_span(span):
    files = {
        'cc_u': f"PIV_planes/Case_CC_Span_{span}.txt_u.csv",
        'sc_u': f"PIV_planes/Case_SC_Span_{span}.txt_u.csv",
        'cc_v': f"PIV_planes/Case_CC_Span_{span}.txt_v.csv",
        'sc_v': f"PIV_planes/Case_SC_Span_{span}.txt_v.csv"
    }
    dfs = {k: _read_as_float(v) for k, v in files.items()}
    if any(v is None for v in dfs.values()):
        print(f"Span {span:02d}: skipped (file missing/empty)")
        return

    # ---------- ΔU ---------------------------------------------------------
    cc_u = _window(dfs['cc_u'])
    sc_u = _window(dfs['sc_u'])
    diff_u = sc_u.values - cc_u.values
    Xdim_u, Y_u = np.meshgrid(convert_sx(cc_u.columns.values), cc_u.index.values)

    _plot(
        Xdim_u, Y_u, diff_u,
        title  = f"Span {span:02d}  ΔU (SC – CC)",
        out_png= os.path.join(OUT_DIR, f"Span_{span:02d}_U.png")
    )

    # ---------- ΔV ---------------------------------------------------------
    cc_v = _window(dfs['cc_v'])
    sc_v = _window(dfs['sc_v'])
    diff_v = sc_v.values - cc_v.values
    Xdim_v, Y_v = np.meshgrid(convert_sx(cc_v.columns.values), cc_v.index.values)

    _plot(
        Xdim_v, Y_v, diff_v,
        title  = f"Span {span:02d}  ΔV (SC – CC)",
        out_png= os.path.join(OUT_DIR, f"Span_{span:02d}_V.png")
    )

# -------------------- batch driver -----------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for span in range(1, 25):
        process_span(span)

if __name__ == "__main__":
    main()
