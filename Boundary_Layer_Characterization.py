import numpy as np
from scipy.integrate import trapz
from u_infinity import compute_u_inf_for_config_span

# Example data arrays (from your dataset)
y = np.array([y1, y2, y3, ...])  # wall-normal coordinates sorted in increasing order
u = np.array([u1, u2, u3, ...])  # corresponding u-velocity values at each y

Ue = 15  # Given free-stream velocity [m/s]

for span in range(1, 25):
    # Use the helper function to get Ue for each span
    Ue = compute_u_inf_for_config_span("CC", span)
    # Compute the displacement thickness for each span
    delta_star = trapz(1 - (u / Ue), y)
    print(f"Span {span}: Displacement thickness δ* (CC) = {delta_star:.4f}")

for span in range(1, 25):
    # Use the helper function to get Ue for each span
    Ue = compute_u_inf_for_config_span("SC", span)
    # Compute the displacement thickness for each span
    delta_star = trapz(1 - (u / Ue), y)
    print(f"Span {span}: Displacement thickness δ* (SC) = {delta_star:.4f}")


# End of Boundary_Layer_Characterization.py