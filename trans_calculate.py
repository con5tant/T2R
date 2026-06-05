import numpy as np
import csv
from scipy.ndimage import gaussian_filter, zoom
import builtins

EXTINCTION_FILE = "atmospheric_extinction_coefficient_cm-1.csv"
MESH_FILES = [("CFD-CHT data/blade", 0.92), ("CFD-CHT data/jishen", 0.92), ("CFD-CHT data/fadongji", 0.85)]
SUBSAMPLE = 5

F_MM = 42.6
F_M = F_MM * 1e-3
DELTA_P_UM = 15.0
DELTA_P_M = DELTA_P_UM * 1e-6
IMAGE_H, IMAGE_W = 512, 512
FOV_DEG = 2.0 * np.degrees(np.arctan((IMAGE_W * DELTA_P_M / 2.0) / F_M))
TAU_OPT = 0.315
G_SYS = 500000.0
T_INT = 0.01

SIGMA_READ = 3.0
SIGMA_DARK = 7.0
SIGMA_PRNU = 0.005
SIGMA_DSNU = 1.0
DEFECT_RATIO = 0.001

T_GROUND = 300.0
T_SKY = 230.0
T_ATM_EFF = 230.0
T_CMB = 2.725

W_GROUND = 0.5

UAV_YAW = 0.0
UAV_PITCH = 0.0
UAV_ROLL = 0.0

SENSOR_POSITION = np.array([0.0, 0.0, -7000.0], dtype=np.float64)


def read_atmospheric_extinction(file_path):
    wl_nm_list, kappa_cm_list = [], []
    with builtins.open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0: continue
            try:
                wl_nm_list.append(float(row[0].strip()))
                kappa_cm_list.append(float(row[1].strip()))
            except (ValueError, IndexError): continue
    wl_um = np.array(wl_nm_list, dtype=np.float64) / 1000.0
    kappa_m = np.array(kappa_cm_list, dtype=np.float64) * 100.0
    return wl_um, kappa_m


def planck_spectral_radiance(wavelength_um, temperature_K, emissivity=1.0):
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    wl = np.asarray(wavelength_um, dtype=np.float64) * 1e-6
    T = np.asarray(temperature_K, dtype=np.float64)
    orig_scalar = (T.ndim == 0)
    if orig_scalar: T = T.reshape(1)
    T2 = T[:, np.newaxis]; wl2 = wl[np.newaxis, :]
    ws = np.maximum(wl2, 1e-10)
    ex = np.clip(h * c / (ws * k * T2), -700, 700)
    L = (2.0 * h * c**2 / (ws**5)) / (np.exp(ex) - 1.0) * 1e-6
    em = np.asarray(emissivity, dtype=np.float64)
    while em.ndim < L.ndim: em = em[..., np.newaxis]
    L = (em * L) / np.pi
    if orig_scalar: L = L[0]
    return L


def load_meshes(mesh_files=None, subsample=SUBSAMPLE):
    if mesh_files is None: mesh_files = MESH_FILES
    all_pts, all_temps, all_nml, all_emiss = [], [], [], []
    for fname, em in mesh_files:
        with builtins.open(fname, 'r') as f: lines = f.readlines()
        processed = [line.replace(',', ' ') for line in lines]
        field = np.loadtxt(processed, skiprows=1)
        field = field[::subsample]
        pts = np.column_stack((field[:, 1], field[:, 2], field[:, 3]))
        temps = field[:, 4]; nr = field[:, 5:8]
        nn = np.linalg.norm(nr, axis=1, keepdims=True)
        nml = nr / np.where(nn == 0, 1, nn)
        print(f"    Loaded '{fname}': {len(pts)} points, eps={em:.3f}")
        all_pts.append(pts); all_temps.append(temps); all_nml.append(nml)
        all_emiss.append(np.full(len(pts), em, dtype=np.float64))
    if len(all_pts) == 0: raise ValueError("No mesh files")
    return (np.vstack(all_pts), np.concatenate(all_temps),
            np.vstack(all_nml), np.concatenate(all_emiss))


def calculate_precise_observation_vectors(camera_center, points):
    v = points - camera_center; n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _rot_matrix(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build_camera_coordinate_system(psi_deg, theta_deg, phi_deg):
    R = _rot_matrix(psi_deg, theta_deg, phi_deg)
    right = R[0, :] / np.linalg.norm(R[0, :])
    down = R[1, :] / np.linalg.norm(R[1, :])
    fwd = R[2, :] / np.linalg.norm(R[2, :]); up = -down
    return SENSOR_POSITION.copy(), [right, up, fwd], fwd


def transform_to_camera_coordinate(points, camera_center, camera_axes):
    right, up, fwd = camera_axes
    return np.dot(points - camera_center, np.array([right, up, fwd]).T)


def _uav_rot_matrix(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def apply_uav_attitude(points, normals):
    R = _uav_rot_matrix(UAV_YAW, UAV_PITCH, UAV_ROLL)
    return np.dot(points, R.T), np.dot(normals, R.T)


def filter_points_by_observation_angle(points, normals, camera_center):
    obs = calculate_precise_observation_vectors(camera_center, points)
    valid = np.sum(-obs * normals, axis=1) > 0
    return points[valid], normals[valid], np.where(valid)[0]


def pinhole_projection(pts_cam, values, f_m=F_M, delta_p_m=DELTA_P_M,
                       image_size=(IMAGE_H, IMAGE_W), zoom_factor=1.0):
    H, W = image_size; u0, v0 = W / 2.0, H / 2.0
    valid = pts_cam[:, 2] > 1e-10
    if not np.any(valid):
        return np.zeros((H, W), dtype=np.float64), np.zeros((H, W), dtype=bool)
    pts = pts_cam[valid]; vals = values[valid]
    u_img = f_m * pts[:, 0] / pts[:, 2]; v_img = -f_m * pts[:, 1] / pts[:, 2]
    if abs(zoom_factor - 1.0) > 1e-12: u_img /= zoom_factor; v_img /= zoom_factor
    col = u_img / delta_p_m + u0; row = v_img / delta_p_m + v0
    col_i = np.round(col).astype(int); row_i = np.round(row).astype(int)
    in_b = (0 <= col_i) & (col_i < W) & (0 <= row_i) & (row_i < H)
    col_i, row_i, z_vals = col_i[in_b], row_i[in_b], pts[in_b, 2]
    vals_in = vals[in_b]
    img_buf = np.full((H, W), np.nan, dtype=np.float64)
    z_buf = np.full((H, W), np.inf)
    for i in range(len(col_i)):
        c, r, z_in, v_in = col_i[i], row_i[i], z_vals[i], vals_in[i]
        if z_in < z_buf[r, c]: z_buf[r, c] = z_in; img_buf[r, c] = v_in
    mask = ~np.isnan(img_buf)
    return np.nan_to_num(img_buf, nan=0.0), mask


def compute_band_integrated_radiance(wl_um, kappa_m, pts, temps, normals,
                                     camera_center, emissivity):
    distances = np.linalg.norm(pts - camera_center, axis=1)
    B_ground = planck_spectral_radiance(wl_um, T_GROUND, emissivity=1.0)
    B_sky = planck_spectral_radiance(wl_um, T_SKY, emissivity=1.0)
    B_env = W_GROUND * B_ground + (1.0 - W_GROUND) * B_sky
    B_atm = planck_spectral_radiance(wl_um, T_ATM_EFF, emissivity=1.0)
    tau = np.exp(-kappa_m[np.newaxis, :] * distances[:, np.newaxis])
    L_path = (1.0 - tau) * B_atm[np.newaxis, :]
    eps_arr = np.asarray(emissivity, dtype=np.float64)
    eps_2d = eps_arr[:, np.newaxis] if eps_arr.ndim == 1 else eps_arr
    L_self = planck_spectral_radiance(wl_um, temps, emissivity=1.0)
    L_self = eps_2d * L_self
    L_ref = (1.0 - eps_2d) * B_env[np.newaxis, :]
    L_leave = L_self + L_ref
    return TAU_OPT * np.trapz(tau * L_leave + L_path, wl_um, axis=1)


def compute_background_radiance(wl_um, kappa_m, distance=10000.0, w_ground=0.0):
    tau = np.exp(-kappa_m * distance)
    B_atm = planck_spectral_radiance(wl_um, T_ATM_EFF, emissivity=1.0)
    L_sky = (1.0 - tau) * B_atm
    B_ground = planck_spectral_radiance(wl_um, T_GROUND, emissivity=1.0)
    L_ground = tau * B_ground
    L_bg_spec = w_ground * L_ground + (1.0 - w_ground) * L_sky
    return float(TAU_OPT * np.trapz(L_bg_spec, wl_um))


def apply_detector_model(rad_map, gain=1.0, offset=0.0, seed=None):
    H, W = rad_map.shape
    rng = np.random.default_rng(seed)
    mu = np.maximum(G_SYS * rad_map * T_INT, 0.0)
    K_prnu = 1.0 + rng.normal(0.0, SIGMA_PRNU, size=(H, W))
    mu_prnu = mu * K_prnu
    shot = rng.poisson(mu_prnu) - mu_prnu
    sigma_rn = np.sqrt(SIGMA_READ**2 + SIGMA_DARK**2)
    rd = rng.normal(0.0, sigma_rn, size=(H, W))
    O_dsnu = rng.normal(0.0, SIGMA_DSNU, size=(H, W))
    sig = mu_prnu + shot + rd + O_dsnu
    dn = gain * sig + offset
    defective = rng.random(size=(H, W)) < DEFECT_RATIO
    dead = defective & (rng.random(size=(H, W)) < 0.5)
    hot = defective & (~dead)
    dn[dead] = 0.0; dn[hot] = 255.0
    return np.clip(np.round(dn), 0, 255).astype(np.uint8)


def crop_norm_square(dn_img, mask, pad=20):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8), np.zeros((IMAGE_H, IMAGE_W), dtype=bool)

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(dn_img.shape[0] - 1, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(dn_img.shape[1] - 1, int(xs.max()) + pad)

    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    half_side = max(box_h, box_w) // 2
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2

    y0 = max(0, cy - half_side)
    y1 = min(dn_img.shape[0] - 1, cy + half_side)
    x0 = max(0, cx - half_side)
    x1 = min(dn_img.shape[1] - 1, cx + half_side)

    actual_h = y1 - y0 + 1
    actual_w = x1 - x0 + 1
    if actual_h < actual_w:
        need = actual_w - actual_h
        if y0 > 0: y0 = max(0, y0 - need)
        elif y1 < dn_img.shape[0] - 1: y1 = min(dn_img.shape[0] - 1, y1 + need)
    elif actual_w < actual_h:
        need = actual_h - actual_w
        if x0 > 0: x0 = max(0, x0 - need)
        elif x1 < dn_img.shape[1] - 1: x1 = min(dn_img.shape[1] - 1, x1 + need)

    ci = dn_img[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    cm = mask[y0:y1 + 1, x0:x1 + 1]

    uav_vals = ci[cm]
    if uav_vals.sum() > 0:
        lo = float(uav_vals.min())
        hi = float(uav_vals.max())
        margin = (hi - lo) * 0.03
        lo -= margin; hi += margin
    else:
        lo = float(ci.min()); hi = float(ci.max())

    if hi > lo:
        ni = np.clip((ci - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    else:
        ni = np.zeros_like(ci, dtype=np.uint8)

    sy = IMAGE_H / ci.shape[0]
    sx = IMAGE_W / ci.shape[1]
    ri = zoom(ni.astype(np.float32), (sy, sx), order=1)
    ri = np.clip(ri, 0, 255).astype(np.uint8)
    rm = zoom(cm.astype(np.float32), (sy, sx), order=0) > 0.5

    return ri, rm


def generate_single_thermal_image(psi=0, theta=0, phi=0, gain=1, offset=0,
                                  blur_sigma=0.0, zoom_factor=1.0, seed=None,
                                  extinction_file=EXTINCTION_FILE, mesh_files=None):
    wl_um, kappa_m = read_atmospheric_extinction(extinction_file)
    bg_radiance = compute_background_radiance(wl_um, kappa_m, distance=10000.0, w_ground=0.0)
    print(f"    BG radiance = {bg_radiance:.6f} W/m2  (FOV={FOV_DEG:.2f}°)")
    pts, temps, nml, emissivities = load_meshes(mesh_files)
    pts, nml = apply_uav_attitude(pts, nml)
    cam_center = SENSOR_POSITION.copy()
    pts_f, nml_f, idx_f = filter_points_by_observation_angle(pts, nml, cam_center)
    I_sen = compute_band_integrated_radiance(wl_um, kappa_m, pts_f, temps[idx_f], nml_f, cam_center, emissivities[idx_f])
    cc, caxes, _ = build_camera_coordinate_system(psi, theta, phi)
    pts_cam = transform_to_camera_coordinate(pts_f, cc, caxes)
    rad_2d, mask = pinhole_projection(pts_cam, I_sen, zoom_factor=zoom_factor)
    rad_2d = np.where(mask, rad_2d, bg_radiance)
    if blur_sigma > 0: rad_2d = gaussian_filter(rad_2d, sigma=blur_sigma, mode='constant')
    dn = apply_detector_model(rad_2d, gain=gain, offset=offset, seed=seed)
    return dn, mask


def generate_figure6_states(psi=0, theta=0, phi=0, blur_sigma=0.0,
                            zoom_factor=1.0, seed=None,
                            extinction_file=EXTINCTION_FILE, mesh_files=None,
                            gain=1.0, offset=0.0):
    wl_um, kappa_m = read_atmospheric_extinction(extinction_file)
    bg_radiance = compute_background_radiance(wl_um, kappa_m, distance=10000.0, w_ground=0.0)
    pts, temps, nml, emissivities = load_meshes(mesh_files)
    pts, nml = apply_uav_attitude(pts, nml)
    cam_center = SENSOR_POSITION.copy()
    pts_f, nml_f, idx_f = filter_points_by_observation_angle(pts, nml, cam_center)
    I_sen = compute_band_integrated_radiance(wl_um, kappa_m, pts_f, temps[idx_f], nml_f, cam_center, emissivities[idx_f])
    cc, caxes, _ = build_camera_coordinate_system(psi, theta, phi)
    pts_cam = transform_to_camera_coordinate(pts_f, cc, caxes)
    rad_map, mask = pinhole_projection(pts_cam, I_sen, zoom_factor=zoom_factor)
    rad_map = np.where(mask, rad_map, bg_radiance)

    # State 1: bare projection (no blur, no noise)
    dn1 = np.clip(G_SYS * rad_map * T_INT, 0, 255).astype(np.uint8)

    # State 2: blur only
    r2 = gaussian_filter(rad_map, sigma=blur_sigma, mode='constant') if blur_sigma > 0 else rad_map.copy()
    dn2 = np.clip(G_SYS * r2 * T_INT, 0, 255).astype(np.uint8)

    # State 3: detector noise only (on unblurred)
    dn3 = apply_detector_model(rad_map, seed=seed, gain=gain, offset=offset)

    # State 4: blur + noise (full pipeline, same as Fig9)
    dn4 = apply_detector_model(r2, seed=seed, gain=gain, offset=offset)

    res = {}
    for sn, dn_img in [("01_projection", dn1), ("02_blur", dn2),
                       ("03_noise", dn3), ("04_final", dn4)]:
        ri, rm = crop_norm_square(dn_img, mask)
        res[sn] = (ri, rm)

    return res