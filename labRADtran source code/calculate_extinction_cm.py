#!/usr/bin/env python3
import numpy as np
import csv
import os

# ===================== 配置项 =====================
INPUT_FILE = "/home/yk/libradtran_test/drone_ir_final.out"
OUTPUT_CSV = "atmospheric_extinction_coefficient_cm-1.csv"
PATH_LENGTH = 7000.0  # 传输路径：6km（向下）
WAVELENGTH_MIN = 3000.0
WAVELENGTH_MAX = 5000.0

# ===================== 读取数据 =====================
def read_radiation_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                print(f"警告：第{line_num+1}行数据不完整，跳过")
                continue
            try:
                wavelength = float(cols[0])
                edir = float(cols[1])
                edn = float(cols[2])
                eup = float(cols[3])
                if WAVELENGTH_MIN <= wavelength <= WAVELENGTH_MAX:
                    data.append([wavelength, edir, edn, eup])
            except ValueError:
                print(f"警告：第{line_num+1}行数据格式错误，跳过")
                continue
    data = np.array(data)
    if len(data) == 0:
        raise ValueError("未读取到有效数据，请检查输入文件路径！")
    return data

# ===================== 严格按比尔-朗伯定律计算消光系数 =====================
def calc_extinction_cm(data, path_len):
    wavelength = data[:, 0]
    # 现在：I0 = 6km 发射强度（eup，向上=无人机自身辐射）
    #      I(L) = 地面接收强度（edir + edn，向下总辐射）
    I0 = data[:, 3]  # 发射端：6km 向上辐射
    I = data[:, 1] + data[:, 2]  # 接收端：地面向下总辐射
    
    # 只过滤极端无效值：I0>0 且 I>0 且 I<I0（符合衰减）
    valid_mask = (I0 > 0) & (I > 0) & (I < I0)
    beta_cm = np.zeros_like(wavelength)
    
    # 核心公式：消光系数(cm⁻¹) = -ln(I/I0) / L * 0.01
    beta_cm[valid_mask] = (-np.log(I[valid_mask] / I0[valid_mask]) / path_len) * 0.01
    
    return np.column_stack([wavelength, beta_cm])

# ===================== 保存结果 =====================
def save_result(result, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["波长(nm)", "大气消光系数(cm⁻¹)"])
        writer.writerows(result)
    print(f"✅ 消光系数文件已生成：{os.path.abspath(output_file)}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        print("🔍 正在读取辐射数据...")
        rad_data = read_radiation_data(INPUT_FILE)
        
        print("🧮 正在计算大气消光系数（cm⁻¹）...")
        extinction_data = calc_extinction_cm(rad_data, PATH_LENGTH)
        
        save_result(extinction_data, OUTPUT_CSV)
        
        print("\n📋 前5行结果：")
        print(f"{'波长(nm)':<12} {'消光系数(cm⁻¹)':<20}")
        for i in range(min(5, len(extinction_data))):
            print(f"{extinction_data[i,0]:<12.1f} {extinction_data[i,1]:<20.6e}")
            
    except Exception as e:
        print(f"❌ 运行失败：{str(e)}")