# Physics-Consistent Synthetic MWIR Imaging of UAVs

A physics-based forward-modeling framework for **synthetic mid-wave infrared (MWIR) imaging** of unmanned aerial vehicles. This repository contains the complete infrared rendering pipeline that converts a three-dimensional UAV thermal state into a sensor-parameterized digital-number (DN) image.

> **Paper:** *"Physics-Consistent Synthetic Mid-Wave Infrared Imaging of Unmanned Aerial Vehicles Using Coupled Flow-Thermal, Radiometric, and Detector-Domain Modeling"*  
> **Authors:** Jian Zhang, Ke Yao, Yali Wang, Zixuan Wang, Zhidong Hu, Yao Zhao  
> **Affiliations:** Northeast Forestry University · Harbin Institute of Technology (Shenzhen) · Army Arms University of PLA

---

## 🔍 Overview

Synthetic infrared imagery is useful for electro-optical evaluation and data generation **only when** the rendered thermal signatures remain physically traceable. This framework links four components into a complete forward chain:

```
CFD-CHT Surface Temperature
    ↓ Atmospheric Radiative Transfer (libRadtran)
    ↓ 3D-to-2D Geometric Projection + Visibility + Z-Buffer
    ↓ Detector-Domain Degradation (blur, noise, non-uniformity, ADC)
    → Sensor-Parameterized MWIR DN Image
```

### Key Contributions

| Contribution | Detail |
|---|---|
| 🔗 **Physics-consistent forward chain** | Full link from aerothermal loading to detector DN output |
| 🛩️ **Full-scale MQ-9-class UAV** | 20.1 m wingspan, steady cruise at 7,600 m, Mach ≈ 0.24 |
| 🌡️ **CFD-CHT thermal prediction** | Temperature-dependent air properties, solar loading, internal nacelle heat source |
| 🌫️ **Atmospheric radiative transfer** | libRadtran line-of-sight transmittance, path radiance, Beer–Lambert extinction |
| 📷 **Detector-domain rendering** | Optical PSF blur, PRNU/DSNU, shot noise, read/dark noise, defective pixels, 8-bit ADC |
| 🔬 **Experimentally constrained** | Detector parameters identified from controlled IR measurements (gain, noise, PSF) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Pillow](https://python-pillow.org/)

```bash
pip install numpy scipy pillow
```

### Required Data Files

The code expects the following directory structure:

```
.
├── atmospheric_extinction_coefficient_cm-1.csv   # wavelength (nm), extinction coefficient (cm⁻¹)
├── CFD-CHT data/
│   ├── blade        # per-vertex data: x, y, z, T, nx, ny, nz (space/comma delimited)
│   ├── jishen       # fuselage mesh
│   └── fadongji     # engine nacelle mesh
├── trans_calculate.py
└── main.py
```

| Component | Emissivity | Material |
|---|---|---|
| Airframe (fuselage, wing, tail) | 0.92 | Carbon-fiber/epoxy composite |
| Engine nacelle / casing | 0.85 | Nickel-based superalloy |

> ⚠️ Mesh data and extinction coefficients are **not included** in this repository. See the paper for data sources or contact the authors.

### Run the Pipeline

```bash
python main.py
```

All generated images, masks, and metrics are written to the `output/` directory.

---

## 📂 Output Structure

```
output/
├── bg_roi.png                        # background ROI mask (for metrics)
├── table4.csv                        # quantitative metrics per view
├── table4.tex                        # LaTeX table
├── images/                           # full 512×512 raw DN images
│   ├── Swoop-by.png
│   ├── Ascend.png
│   └── Cruise.png
├── masks/                            # binary target masks
│   ├── Swoop-by.png
│   ├── Ascend.png
│   └── Cruise.png
├── fig6_states/                      # four processing stages per view
│   ├── <View>_01_projection.png      # bare radiometric projection
│   ├── <View>_02_blur.png            # optical blur only
│   ├── <View>_03_noise.png           # detector noise only
│   ├── <View>_04_final.png           # full pipeline (blur + noise + quantization)
└── fig9_views/                       # cropped & intensity-normalized views
    ├── <View>_fig9.png
    └── <View>_fig9_mask.png
```

---

## ⚙️ Configuration Reference

### Sensor Parameters (`trans_calculate.py`)

| Parameter | Value | Description |
|---|---|---|
| **Focal length** `F_MM` | 42.6 mm | Effective focal length |
| **Pixel pitch** `DELTA_P_UM` | 15.0 µm | Detector pixel pitch |
| **Resolution** `IMAGE_H × IMAGE_W` | 512 × 512 | Focal-plane array size |
| **Field of view** `FOV_DEG` | ~10.3° × 10.3° | Diagonal ≈ 14.6° |
| **Optical transmission** `TAU_OPT` | 0.315 | Effective MWIR optics transmittance |
| **System gain** `G_SYS` | 500,000 e⁻/W | Effective conversion factor |
| **Integration time** `T_INT` | 0.01 s | Exposure time |
| **Observation distance** | 7,000 m | Baseline slant range |
| **MWIR band** | 3.0–5.0 µm | Spectral integration range |

### Detector Noise Parameters

| Symbol | Value | Description |
|---|---|---|
| σ<sub>read</sub> | 3.0 e⁻ | Read noise (RMS) |
| σ<sub>dark</sub> | 7.0 e⁻ | Dark current noise (RMS) |
| σ<sub>PRNU</sub> | 0.5% | Pixel response non-uniformity |
| σ<sub>DSNU</sub> | 1.0 e⁻ | Dark-signal non-uniformity |
| Defect ratio | 0.1% | Fraction of dead/hot pixels |

### Environmental Temperatures

| Parameter | Value | Description |
|---|---|---|
| T<sub>ground</sub> | 300 K | Ground-equivalent temperature |
| T<sub>sky</sub> | 230 K | Clear-sky equivalent temperature |
| T<sub>atm,eff</sub> | 230 K | Effective atmospheric path temperature |
| T<sub>CMB</sub> | 2.725 K | Cosmic microwave background |

### View Configurations (`main.py`)

Three observation aspects are pre-defined:

| View | Obs. Angle | Azimuth | Camera Roll | UAV Yaw | UAV Pitch | UAV Roll |
|---|---|---|---|---|---|---|
| **Swoop-by** | −70° | 0° | 180° | 0° | −80° | 0° |
| **Ascend** | −70° | 120° | 220° | 0° | 0° | 0° |
| **Cruise** | −70° | 120° | 180° | 0° | 0° | 0° |

---

## 🧪 Pipeline Details

### Stage 1 — Aerothermal Prediction (CFD-CHT)

The surface temperature field \(T_{\text{skin}}\) is computed using a steady CFD-CHT model in ANSYS Fluent:

- **Grid**: Hybrid mesh, ~659M cells (medium), \(y^+_{\text{target}} = 60\)
- **Turbulence**: Realizable \(k\)-\(\varepsilon\) with standard wall functions
- **Air properties**: Temperature-dependent (Sutherland's law for μ and k)
- **Radiation**: DO model + solar ray tracing (solar altitude 73.5°, ~1,100 W/m²)
- **Internal heat**: Uniform volumetric source 9 × 10⁴ W/m³ in nacelle (≈ 37 kW waste heat)

> **Sensitivity findings:** Constant-property modeling overpredicts peak skin temperature by ≈ 8 K; solar irradiation produces local increases of up to 9 K.

### Stage 2 — Radiometric Modeling

**Planck's law** (spectral blackbody radiance):

$$
B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5} \left[ \exp\!\left(\frac{h c}{\lambda k_B T}\right) - 1 \right]^{-1}
$$

**Surface-leaving radiance** (gray-body emission + reflection):

$$
L_{\text{leave},i}(\lambda) = \epsilon_i \, B_\lambda(\lambda, T_{\text{skin},i}) + (1 - \epsilon_i) \, L_{\text{env}}(\lambda)
$$

where \(L_{\text{env}}\) is the weighted sum of ground and sky radiance.

**Atmospheric propagation** (Beer–Lambert + path radiance):

$$
L_{\text{sensor},i}(\lambda) = \tau(\lambda, d_i) \, L_{\text{leave},i}(\lambda) + \bigl[1 - \tau(\lambda, d_i)\bigr] \, B_\lambda(\lambda, T_{\text{atm,eff}})
$$

Atmospheric transmittance \(\tau(\lambda, d)\) is pre-computed with **libRadtran** using ISA clear-sky profile at 7,600 m.

**Band-integrated sensor input:**

$$
I_{\text{sensor},i} = \tau_{\text{opt}} \int_{\lambda_1}^{\lambda_2} L_{\text{sensor},i}(\lambda) \, d\lambda
$$

### Stage 3 — Geometric Projection

- **Coordinate system**: Right-Down-Forward (RDF) sensor frame  
- **Projection**: Pinhole model with focal length \(f = 42.6\) mm  
- **Visibility**: Back-face culling (\(\mathbf{n}_i \cdot \mathbf{v}_{\text{obs},i} > 0\))  
- **Occlusion**: Z-buffer (depth test per pixel)  
- **Background**: Constant radiance from atmospheric path + ground emission at 10 km

### Stage 4 — Detector-Domain Rendering

The ideal radiometric image is degraded by:

1. **Optical blur** — Gaussian PSF convolution (\(\sigma = 1.059\) pixels, from knife-edge measurement)
2. **PRNU** — Pixel-wise gain variation ∼ 𝒩(1, 0.005²)
3. **Shot noise** — Poisson-distributed photoelectron noise
4. **Read + dark noise** — Gaussian 𝒩(0, σ²), σ = √(3² + 7²) ≈ 7.6 e⁻
5. **DSNU** — Pixel-wise offset ∼ 𝒩(0, 1.0²)
6. **Defective pixels** — 0.1% random dead (DN = 0) / hot (DN = 255)
7. **ADC quantization** — 8-bit, clipped to [0, 255]

---

## 📊 Output Metrics (Table 4)

| Metric | Formula / Description |
|---|---|
| \(G_{\max}\) | Peak DN on target pixels |
| \(\bar{G}_t\) | Mean DN over target mask |
| \(\bar{G}_{\text{bg}}\) | Mean DN over background ROI |
| **CTB** | \(\bar{G}_t - \bar{G}_{\text{bg}}\) (contrast-to-background) |
| \(A_{\text{hot}}\) | Number of pixels above threshold \(\mu_{\text{bg}} + 3\sigma_{\text{bg}}\) |
| \((u_{\text{hot}}, v_{\text{hot}})\) | Centroid of hot-spot cluster |

---

## 📄 License

This project is provided under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- This work uses **libRadtran** for atmospheric radiative transfer calculations.
- CFD-CHT simulations were performed using **ANSYS Fluent**.
- The target platform is based on the public geometry of an MQ-9-class MALE UAV.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.

