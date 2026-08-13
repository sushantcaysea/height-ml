"""
measure_body.py
---------------
Turn SMPLer-X .npz outputs into height / chest / waist (cm).

MATH IDEA (simple):
  - A body mesh = many 3D points (x, y, z), each in meters.
  - HEIGHT  = highest point - lowest point  (along up-axis Y)
  - CHEST / WAIST = "cut" the body with a horizontal plane,
                    look at the ring of points on that cut,
                    measure how long that ring is (circumference).

  Circumference of a closed ring of points p0, p1, ..., pn:
      length = |p1-p0| + |p2-p1| + ... + |p0-pn|
  where |a-b| is Euclidean distance = sqrt((x2-x1)^2 + (z2-z1)^2)
"""

import os
import glob
import sys
import numpy as np
import torch

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "common", "utils"))
import smplx  # library that builds a 3D body from shape numbers (betas)


def circumference_at_height(vertices, y_level, band=0.02):
    """
    Measure "tape around the body" at one height.

    vertices : (N, 3) array
        Each row is one mesh point [x, y, z] in meters.
        In SMPL-X T-pose: Y = up (feet low, head high).

    y_level : float
        The height (Y value) where we pretend to wrap the tape.

    band : float
        We cannot cut infinitely thin. Keep points whose Y is close:
            |y_i - y_level| < band
        Default band = 0.02 m = 2 cm thick slice.

    Returns circumference in meters.
    """

    # --- Step A: take a thin horizontal slice of the body ---
    # Boolean mask: True for points near the chosen height
    # abs(y - y_level) < band  <=>  point is inside the slice
    mask = np.abs(vertices[:, 1] - y_level) < band
    # Project those 3D points onto the ground plane (drop Y):
    # keep only (x, z) so we get a 2D outline looking from above
    ring = vertices[mask][:, [0, 2]]  # shape (M, 2)

    # If too few points, widen the slice (2.5 * band)
    if len(ring) < 8:
        mask = np.abs(vertices[:, 1] - y_level) < band * 2.5
        ring = vertices[mask][:, [0, 2]]
    if len(ring) < 8:
        return float("nan")  # not enough points to measure

    # --- Step B: outline length (best method = convex hull) ---
    # Convex hull = smallest convex shape wrapping all ring points
    # (like a rubber band around nails on a board).
    # Walking around hull edges ≈ tape measure around the torso.
    if ConvexHull is not None:
        try:
            hull = ConvexHull(ring)
            # hull.vertices = indices of points on the outer outline, in order
            pts = ring[hull.vertices]
            # Close the loop: add first point again at the end
            # so last segment connects back to start
            pts = np.vstack([pts, pts[0]])

            # Distance between consecutive points:
            #   ||pts[i+1] - pts[i]||  = sqrt(dx^2 + dz^2)
            # Sum = full perimeter
            edge_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            return float(np.sum(edge_lengths))
        except Exception:
            pass

    # --- Step C: fallback if hull fails ---
    # Pretend the cross-section is an ellipse with:
    #   width  = max(x) - min(x)   -> semi-axis a = width/2
    #   depth  = max(z) - min(z)   -> semi-axis b = depth/2
    # Ramanujan approximation for ellipse perimeter:
    #   P ≈ π [ 3(a+b) - sqrt( (3a+b)(a+3b) ) ]
    width = ring[:, 0].max() - ring[:, 0].min()
    depth = ring[:, 1].max() - ring[:, 1].min()
    a, b = width / 2.0, depth / 2.0
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def measure_vertices(vertices):
    """
    Compute the 3 assignment measurements from one mesh.

    HEIGHT math:
        height = max(y) - min(y)
        (top of head minus bottom of feet, in meters)

    CHEST / WAIST math:
        Choose a height as a fraction of body height from the feet:
            chest_y = y_min + 0.74 * height   # ~ chest / bust line
            waist_y = y_min + 0.58 * height   # ~ waist / navel line
        Then call circumference_at_height at those Y values.

    Finally convert m -> cm by * 100.
    """
    y_min = vertices[:, 1].min()  # feet (lowest)
    y_max = vertices[:, 1].max()  # head (highest)
    height_m = float(y_max - y_min)

    # Fractions 0.74 and 0.58 are approximate anatomy landmarks
    # (not exact medical definitions — good enough for this mini-pipeline)
    chest_m = circumference_at_height(vertices, y_min + 0.74 * height_m)
    waist_m = circumference_at_height(vertices, y_min + 0.58 * height_m)

    return {
        "height_cm": height_m * 100,  # 1 m = 100 cm
        "chest_cm": chest_m * 100,
        "waist_cm": waist_m * 100,
    }


def build_tpose_vertices(betas, model):
    """
    Build a standing T-pose body from shape only.

    betas : shape coefficients from SMPLer-X (.npz file)
        Think: 10 numbers that stretch the average body
        (taller/shorter, wider/thinner, etc.).
        SMPL-X:  mesh ≈ mean_body + sum_i (betas[i] * shape_basis[i])
        (skinning/joints also applied inside smplx; we zero the pose.)

    Why T-pose?
        Posed mesh (arms out, bent) makes waist slices wrong.
        Zero pose = arms in neutral T → cleaner chest/waist rings.

    Returns vertices as NumPy array shape (10475, 3) for SMPL-X.
    """
    # Make betas a torch tensor of shape (1, 10)
    betas_t = torch.tensor(betas, dtype=torch.float32).view(1, -1)
    if betas_t.shape[1] < 10:
        # pad with zeros if fewer than 10 coeffs
        pad = torch.zeros(1, 10 - betas_t.shape[1])
        betas_t = torch.cat([betas_t, pad], dim=1)
    betas_t = betas_t[:, :10]

    # All pose parameters = 0  =>  neutral / T-pose
    # body_pose (1, 63): 21 joints * 3 (axis-angle x,y,z)
    out = model(
        betas=betas_t,
        body_pose=torch.zeros(1, 63),
        global_orient=torch.zeros(1, 3),
        left_hand_pose=torch.zeros(1, 45),
        right_hand_pose=torch.zeros(1, 45),
        jaw_pose=torch.zeros(1, 3),
        leye_pose=torch.zeros(1, 3),
        reye_pose=torch.zeros(1, 3),
        expression=torch.zeros(1, 10),
        transl=torch.zeros(1, 3),
    )
    # out.vertices: (1, N, 3) on GPU/CPU tensor -> NumPy (N, 3)
    return out.vertices[0].detach().cpu().numpy()


def main():
    # --- paths ---
    smplx_dir = os.path.join(ROOT, "demo", "results", "my_person", "smplx")
    model_path = os.path.join(ROOT, "common", "utils", "human_model_files")
    npz_files = sorted(glob.glob(os.path.join(smplx_dir, "*.npz")))
    if not npz_files:
        print("No .npz files found in", smplx_dir)
        return

    # --- load the SMPL-X "blank doll" once ---
    print("Loading SMPL-X body model...")
    model = smplx.create(
        model_path,
        model_type="smplx",
        gender="neutral",
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True,
    )
    model.eval()  # inference mode (no training)

    print("Body measurements (T-pose from predicted shape betas)")
    print("Units: centimeters\n")
    print(f"{'file':<16} {'height_cm':>10} {'chest_cm':>10} {'waist_cm':>10}")
    print("-" * 50)

    rows = []
    for path in npz_files:
        # .npz holds: betas, poses, transl, ...
        # We only need betas (shape) for T-pose measuring
        data = np.load(path)
        betas = data["betas"]

        # betas -> 3D points -> height/chest/waist
        verts = build_tpose_vertices(betas, model)
        m = measure_vertices(verts)

        name = os.path.basename(path)
        print(f"{name:<16} {m['height_cm']:10.1f} {m['chest_cm']:10.1f} {m['waist_cm']:10.1f}")
        rows.append(m)

    # Average across photos:
    #   avg = (m1 + m2 + ... + mK) / K
    avg_h = float(np.nanmean([r["height_cm"] for r in rows]))
    avg_c = float(np.nanmean([r["chest_cm"] for r in rows]))
    avg_w = float(np.nanmean([r["waist_cm"] for r in rows]))
    print("-" * 50)
    print(f"{'AVERAGE':<16} {avg_h:10.1f} {avg_c:10.1f} {avg_w:10.1f}")
    print("\nUse AVERAGE in your report; compare to tape measurements for validation.")


if __name__ == "__main__":
    main()
