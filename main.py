#!/usr/bin/env python3
import builtins
import os, sys, csv, numpy as np
from PIL import Image
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")
import trans_calculate as ir

SIGMA_PSF   = 1.059
NOISE_LEVEL = 0.2
GRID_SIZE   = (512, 512)
BG_PADDING  = 20
D_OBS       = 7000.0
ZOOM_FACTOR = 0.015

VIEWS = [
    ("Swoop-by",    -70,    0,     180,   0,   -80,   0),
    ("Ascend",      -70,    120,   220,   0,   0,   0),
    ("Cruise",      -70,    120,   180,   0,   0,   0),
]


def sensor_pos_from_obs_angle(alpha_deg, beta_deg=0.0):
    a = np.radians(alpha_deg); b = np.radians(beta_deg)
    z = -D_OBS; sin_a, cos_a = np.sin(a), np.cos(a)
    r_horiz = 1e8 if abs(sin_a) < 1e-12 else D_OBS * cos_a / sin_a
    return np.array([-r_horiz * np.sin(b), -r_horiz * np.cos(b), z], dtype=np.float64)


def compute_metrics(img, tgt_mask, bg_mask, tf=3.0):
    t = img[tgt_mask].astype(np.float32)
    b = img[bg_mask].astype(np.float32) if bg_mask.sum() > 0 else img[~tgt_mask].astype(np.float32)
    th = b.mean() + tf * float(b.std())
    hp = tgt_mask & (img.astype(np.float32) > th)
    ys, xs = np.where(hp)
    return {"Gmax": round(float(t.max()), 2), "Gt_mean": round(float(t.mean()), 2),
            "Gbg_mean": round(float(b.mean()), 2), "CTB": round(float(t.mean()) - float(b.mean()), 2),
            "SCR": "N/A", "Ahot": int(hp.sum()),
            "uhot": round(float(xs.mean()), 1) if hp.sum() > 0 else 0.0,
            "vhot": round(float(ys.mean()), 1) if hp.sum() > 0 else 0.0}


def compute_euler_from_fwd(fwd, roll_deg=0.0):
    theta = np.degrees(np.arcsin(np.clip(-fwd[0], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(fwd[1], fwd[2]))
    return roll_deg, theta, phi


def generate_view(view_name, obs_angle, azimuth, roll_cam,
                  uav_yaw, uav_pitch, uav_roll, out_dir):
    img_dir = os.path.join(out_dir, "images", "")
    msk_dir = os.path.join(out_dir, "masks", "")
    fig6_dir = os.path.join(out_dir, "fig6_states", "")
    fig9_dir = os.path.join(out_dir, "fig9_views", "")
    for d in [img_dir, msk_dir, fig6_dir, fig9_dir]:
        os.makedirs(d, exist_ok=True)

    sp = sensor_pos_from_obs_angle(obs_angle, azimuth)
    ir.SENSOR_POSITION = sp
    ir.UAV_YAW = uav_yaw; ir.UAV_PITCH = uav_pitch; ir.UAV_ROLL = uav_roll
    pos_norm = np.linalg.norm(sp); fwd = -sp / pos_norm
    psi, theta, phi = compute_euler_from_fwd(fwd, roll_cam)

    print(f"\n  --- {view_name} ---")
    print(f"  obs_angle={obs_angle}°, azimuth={azimuth}°, roll_cam={roll_cam}°")
    print(f"  UAV: yaw={uav_yaw}°, pitch={uav_pitch}°, roll={uav_roll}°")
    print(f"  sensor=({sp[0]:.1f},{sp[1]:.1f},{sp[2]:.1f}) |sp|={pos_norm:.1f}m")

    kw = dict(psi=psi, theta=theta, phi=phi, zoom_factor=ZOOM_FACTOR)

    print(f"  Figure 6: 4 states (shared normalization)...")
    states = ir.generate_figure6_states(blur_sigma=SIGMA_PSF, **kw, gain=1)
    for sn, (im, _) in states.items():
        Image.fromarray(im).save(os.path.join(fig6_dir, f"{view_name}_{sn}.png"))

    print(f"  Full pipeline...")
    dn, mask = ir.generate_single_thermal_image(blur_sigma=SIGMA_PSF, **kw, gain=1, offset=0)
    Image.fromarray(dn).save(os.path.join(img_dir, f"{view_name}.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(msk_dir, f"{view_name}.png"))
    print(f"    DN range [{dn.min()},{dn.max()}]")

    ci, cm = ir.crop_norm_square(dn, mask)
    Image.fromarray(ci).save(os.path.join(fig9_dir, f"{view_name}_fig9.png"))
    Image.fromarray((cm * 255).astype(np.uint8)).save(os.path.join(fig9_dir, f"{view_name}_fig9_mask.png"))

    ys, xs = np.where(mask)
    if len(ys) > 0:
        y0 = max(0, int(ys.min()) - BG_PADDING)
        y1 = min(dn.shape[0] - 1, int(ys.max()) + BG_PADDING)
        x0 = max(0, int(xs.min()) - BG_PADDING)
        x1 = min(dn.shape[1] - 1, int(xs.max()) + BG_PADDING)
        bbox = (y0, y1, x0, x1)
    else:
        bbox = (0, 0, 0, 0)

    return (view_name, ci, cm, bbox)


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"  views={len(VIEWS)}")

    bg = np.zeros((512, 512), dtype=np.uint8)
    bg[30:130, 30:130] = 255
    bg_path = os.path.join(out_dir, "bg_roi.png")
    Image.fromarray(bg).save(bg_path)

    fig9_data = []
    for vn, oa, az, rc, uy, up, ur in VIEWS:
        r = generate_view(vn, oa, az, rc, uy, up, ur, out_dir)
        if r:
            fig9_data.append(r)

    if fig9_data:
        print("\n[Table 4]")
        bg_f = np.asarray(Image.open(bg_path).convert("L"), dtype=np.uint8) > 128
        rows = []
        for vn, ci, cm, (y0, y1, x0, x1) in fig9_data:
            h_in = y1 - y0 + 1; w_in = x1 - x0 + 1
            half_side = max(h_in, w_in) // 2
            cy = (y0 + y1) // 2; cx = (x0 + x1) // 2
            by0 = max(0, cy - half_side); by1 = min(511, cy + half_side)
            bx0 = max(0, cx - half_side); bx1 = min(511, cx + half_side)
            ah = by1 - by0 + 1; aw = bx1 - bx0 + 1
            if ah < aw:
                need = aw - ah
                if by0 > 0: by0 = max(0, by0 - need)
                elif by1 < 511: by1 = min(511, by1 + need)
            elif aw < ah:
                need = ah - aw
                if bx0 > 0: bx0 = max(0, bx0 - need)
                elif bx1 < 511: bx1 = min(511, bx1 + need)
            bc = bg_f[by0:by1 + 1, bx0:bx1 + 1]
            sy = 512.0 / bc.shape[0]; sx = 512.0 / bc.shape[1]
            br = zoom(bc.astype(np.float32), (sy, sx), order=0) > 0.5
            m = compute_metrics(ci, cm, br)
            m["view"] = vn; rows.append(m)
            print(f"  {vn}: Gmax={m['Gmax']:.1f} Gt={m['Gt_mean']:.1f} Gbg={m['Gbg_mean']:.1f} CTB={m['CTB']:.1f}")

        fields = ["view", "Gmax", "Gt_mean", "Gbg_mean", "CTB", "SCR", "Ahot", "uhot", "vhot"]
        csv_path = os.path.join(out_dir, "table4.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

        order = ["Swoop-by", "Ascend", "Cruise"]
        rm = {r["view"]: r for r in rows}
        tex = ["\\begin{table}[htbp]\\centering",
               "\\caption{Quantitative metrics.}", "\\label{tab:metrics}",
               "\\begin{tabular}{lccccccc}\\toprule",
               "View & $G_{\\max}$ & $\\bar{G}_t$ & $\\bar{G}_{\\mathrm{bg}}$ & CTB & $A_{\\mathrm{hot}}$ & $(u_{\\mathrm{hot}}, v_{\\mathrm{hot}})$ \\\\ \\midrule"]
        for v in order:
            if v in rm:
                r_ = rm[v]
                tex.append(f"  {v} & {r_['Gmax']:.2f} & {r_['Gt_mean']:.2f} & {r_['Gbg_mean']:.2f} & {r_['CTB']:.2f} & {r_['Ahot']} & ({r_['uhot']:.0f}, {r_['vhot']:.0f}) \\\\")
        tex.append("\\bottomrule\\end{tabular}\\end{table}")
        with open(os.path.join(out_dir, "table4.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex))

    print(f"\n{'='*60}\n Complete.\n{'='*60}")


if __name__ == "__main__":
    main()
