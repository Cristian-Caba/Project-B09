#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# ΔU and ΔV (SC − CC) contour maps for Span_1‥24
# x∈[120,155] mm, y∈[0,4] mm ;  X-axis nondimensionalised with convert_sx
# Colour scale fixed:  –8  (blue)  …  0  (white)  …  +3.5  (red)
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from dimensionalization import convert_sx

# ---- constants -------------------------------------------------------------
X_MIN, X_MAX = 120, 155           # mm
Y_MIN, Y_MAX =   0,   4           # mm
N_LEVELS     = 256
OUT_DIR      = "images_diff"
NORM         = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=3.5)

# ---- helpers ---------------------------------------------------------------
def _read_as_float(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df.index   = df.index.astype(float)    # Y
    df.columns = df.columns.astype(float)  # X
    return df

def _window(df):
    return df.loc[
        (df.index   >= Y_MIN) & (df.index   <= Y_MAX),
        (df.columns >= X_MIN) & (df.columns <= X_MAX)
    ]

# ---------------------------------------------------------------------------
def _plot(Xdim, Y, diff, title, file_png):
    """Contour plot with fixed −8 … +3.5 blue-white-red scale."""
    norm   = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=3.5)
    levels = np.linspace(-8, 3.5, 257)      # ← 256 equal bands, fixed range
    plt.figure()
    plt.contourf(
        Xdim, Y, diff,
        levels=levels,            # <<––––  explicit full-range levels
        cmap='bwr',
        norm=norm,
        extend='both'
    )
    plt.colorbar(label="SC – CC  (m/s)")
    plt.xlabel("Dimensionless X (x/c)")
    plt.ylabel("Y [mm]")
    plt.title(title)
    plt.savefig(file_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_png}")
# ---------------------------------------------------------------------------

# ---- one span --------------------------------------------------------------
def process_span(span_idx):
    cc_u = f"PIV_planes/Case_CC_Span_{span_idx}.txt_u.csv"
    sc_u = f"PIV_planes/Case_SC_Span_{span_idx}.txt_u.csv"
    cc_v = f"PIV_planes/Case_CC_Span_{span_idx}.txt_v.csv"
    sc_v = f"PIV_planes/Case_SC_Span_{span_idx}.txt_v.csv"

    # ---------- U -----------------------------------------------------------
    cc_u_sub = _window(_read_as_float(cc_u))
    sc_u_sub = _window(_read_as_float(sc_u))
    diff_u   = sc_u_sub.values - cc_u_sub.values
    Xdim_u, Y_u = np.meshgrid(
        convert_sx(cc_u_sub.columns.values),
        cc_u_sub.index.values
    )
    _plot(
        Xdim_u, Y_u, diff_u,
        title=f"Span {span_idx:02d}  ΔU (SC – CC)",
        file_png=os.path.join(OUT_DIR, f"Span_{span_idx:02d}_U.png")
    )

    # ---------- V -----------------------------------------------------------
    cc_v_sub = _window(_read_as_float(cc_v))
    sc_v_sub = _window(_read_as_float(sc_v))
    diff_v   = sc_v_sub.values - cc_v_sub.values
    Xdim_v, Y_v = np.meshgrid(
        convert_sx(cc_v_sub.columns.values),
        cc_v_sub.index.values
    )
    _plot(
        Xdim_v, Y_v, diff_v,
        title=f"Span {span_idx:02d}  ΔV (SC – CC)",
        file_png=os.path.join(OUT_DIR, f"Span_{span_idx:02d}_V.png")
    )

# ---- batch driver ----------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for span in range(1, 25):
        process_span(span)

if __name__ == "__main__":
    main()
