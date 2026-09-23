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


def torso_only_ring(ring, x_limit=None):
    """
    Keep only torso points from a horizontal (x, z) slice.

    T-pose arms stick out in X (often |x| > 0.3 m). A real chest half-width
    is ~12–22 cm. We always clip |x| so the convex hull cannot wrap the arms.
    """
    if len(ring) < 8:
        return ring

    cx = float(np.median(ring[:, 0]))
    if x_limit is None or not np.isfinite(x_limit) or x_limit <= 0:
        x_limit = 0.17
    x_limit = float(np.clip(x_limit, 0.12, 0.20))
    torso = ring[np.abs(ring[:, 0] - cx) <= x_limit]
    return torso if len(torso) >= 8 else ring


def _ellipse_perimeter(ring):
    """Smooth tape length: PCA ellipse around the torso slice (Ramanujan)."""
    c = ring.mean(axis=0)
    pts = ring - c
    cov = np.cov(pts.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None
    evals, evecs = np.linalg.eigh(cov)
    aligned = pts @ evecs
    a = 0.5 * (aligned[:, 0].max() - aligned[:, 0].min())
    b = 0.5 * (aligned[:, 1].max() - aligned[:, 1].min())
    if a <= 1e-6 or b <= 1e-6:
        return None
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def _hull_perimeter(ring):
    if ConvexHull is None or len(ring) < 8:
        return None
    try:
        hull = ConvexHull(ring)
        pts = ring[hull.vertices]
        pts = np.vstack([pts, pts[0]])
        return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))
    except Exception:
        return None


def ring_circumference(ring):
    """
    Tape around the slice. Convex hull of mesh vertices is slightly jagged
    and tends to overestimate; a fitted ellipse is smoother. Use the smaller
    of the two when both exist (closer to a real tape).
    """
    hull = _hull_perimeter(ring)
    ell = _ellipse_perimeter(ring)
    vals = [v for v in (hull, ell) if v is not None and np.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    return float(min(vals))


def circumference_at_height(vertices, y_level, band=0.012, x_limit=None):
    """
    Measure "tape around the body" at one height.

    vertices : (N, 3) array
        Each row is one mesh point [x, y, z] in meters.
        In SMPL-X T-pose: Y = up (feet low, head high).

    y_level : float
        The height (Y value) where we pretend to wrap the tape.

    band : float
        Keep points with |y - y_level| < band. Default 1.2 cm (thin slice).

    x_limit : float or None
        Max |x| from the body midline (meters). Drops arms.

    Returns circumference in meters.
    """
    mask = np.abs(vertices[:, 1] - y_level) < band
    ring = vertices[mask][:, [0, 2]]

    if len(ring) < 8:
        mask = np.abs(vertices[:, 1] - y_level) < band * 2.5
        ring = vertices[mask][:, [0, 2]]
    if len(ring) < 8:
        return float("nan")

    ring = torso_only_ring(ring, x_limit=x_limit)
    return ring_circumference(ring)


def _shoulder_half_width(joints):
    """SMPL-X joints: 16 = left shoulder, 17 = right shoulder."""
    if joints is None or len(joints) <= 17:
        return None
    half = min(abs(float(joints[16, 0])), abs(float(joints[17, 0])))
    if not np.isfinite(half) or half < 0.05:
        return None
    return half


def _slice_heights(vertices, joints, y_min, height_m):
    """
    Chest / waist Y from SMPL-X joints when possible.
    0 pelvis, 3 spine1, 6 spine2, 9 spine3, 12 neck, 16/17 shoulders.
    """
    chest_y = y_min + 0.71 * height_m
    waist_y = y_min + 0.545 * height_m
    if joints is None or len(joints) <= 17:
        return chest_y, waist_y
    try:
        pelvis = float(joints[0, 1])
        spine1 = float(joints[3, 1])
        spine3 = float(joints[9, 1])
        neck = float(joints[12, 1]) if len(joints) > 12 else spine3
        sh_y = 0.5 * (float(joints[16, 1]) + float(joints[17, 1]))
        # Bust: below shoulders / around spine3, not up in the armpits
        chest_y = 0.50 * spine3 + 0.30 * sh_y + 0.20 * neck
        # Natural waist: between pelvis and spine1 (navel), not lower rib
        waist_y = 0.42 * pelvis + 0.58 * spine1
        chest_y = float(np.clip(chest_y, y_min + 0.62 * height_m, y_min + 0.78 * height_m))
        waist_y = float(np.clip(waist_y, y_min + 0.50 * height_m, y_min + 0.62 * height_m))
    except Exception:
        pass
    return chest_y, waist_y


def measure_vertices(vertices, joints=None):
    """
    Compute the 3 assignment measurements from one mesh.

    HEIGHT: max(y) - min(y) on the T-pose mesh.
    CHEST / WAIST: torso-only tape at joint-based heights.
    """
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    height_m = float(y_max - y_min)

    sh = _shoulder_half_width(joints)
    x_chest = 0.80 * sh if sh else 0.16
    x_waist = 0.64 * sh if sh else 0.14
    chest_y, waist_y = _slice_heights(vertices, joints, y_min, height_m)

    chest_m = circumference_at_height(vertices, chest_y, x_limit=x_chest)
    waist_m = circumference_at_height(vertices, waist_y, x_limit=x_waist)

    return {
        "height_cm": height_m * 100,
        "chest_cm": chest_m * 100,
        "waist_cm": waist_m * 100,
    }


def build_tpose_vertices(betas, model, return_joints=False):
    """
    Build a standing T-pose body from shape only.

    betas : shape coefficients from SMPLer-X (.npz file)
        Think: 10 numbers that stretch the average body
        (taller/shorter, wider/thinner, etc.).
        SMPL-X:  mesh ≈ mean_body + sum_i (betas[i] * shape_basis[i])
        (skinning/joints also applied inside smplx; we zero the pose.)

        Why T-pose?
        Posed mesh (arms out, bent) makes waist slices wrong.
        Zero pose = arms in a known T. We then drop arm points and
        measure only the torso ring.

    Returns vertices as NumPy array shape (10475, 3) for SMPL-X.
    If return_joints is True, also returns joints (J, 3).
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
    verts = out.vertices[0].detach().cpu().numpy()
    if return_joints:
        joints = out.joints[0].detach().cpu().numpy()
        return verts, joints
    return verts


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
        verts, joints = build_tpose_vertices(betas, model, return_joints=True)
        m = measure_vertices(verts, joints)

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
