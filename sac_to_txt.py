#!/usr/bin/env python3
"""
QuakeWatch - SAC 轉 TXT 格式
將目錄下所有 .SAC 檔案轉換為 txt 格式
格式: {13位時間戳} {counts}
"""

import os
import glob
from obspy import read
import numpy as np

# ============================================================
# 設定參數
# ============================================================
SCALE_FACTOR = 10000  # 浮點數轉 counts 的倍數 (例如: 10000)
# ============================================================

def convert_sac_to_txt(sac_file):
    """
    將單個 SAC 檔案轉換為 txt 格式

    Args:
        sac_file: SAC 檔案路徑

    Returns:
        txt_file: 輸出的 txt 檔案路徑
    """
    try:
        # 讀取 SAC 檔案
        st = read(sac_file)
        tr = st[0]  # SAC 檔案只包含一個 trace

        # 取得資料和時間資訊
        data = tr.data
        start_time = tr.stats.starttime
        sampling_rate = tr.stats.sampling_rate
        npts = tr.stats.npts

        # 計算每個資料點的時間戳 (13位毫秒時間戳)
        timestamps = []
        for i in range(npts):
            # 計算每個資料點的絕對時間
            point_time = start_time + (i / sampling_rate)
            # 轉換為 13 位毫秒時間戳
            timestamp_ms = int(point_time.timestamp * 1000)
            timestamps.append(timestamp_ms)

        # 轉換 counts (SAC 資料是浮點數，需要乘以倍數)
        counts = np.round(data * SCALE_FACTOR).astype(np.int32)

        # 生成輸出檔案名稱
        base_name = os.path.splitext(sac_file)[0]
        txt_file = f"{base_name}.txt"

        # 寫入 txt 檔案
        with open(txt_file, 'w') as f:
            for timestamp, count in zip(timestamps, counts):
                f.write(f"{timestamp} {count}\n")

        return txt_file, npts

    except Exception as e:
        print(f"✗ 轉換失敗: {sac_file}")
        print(f"  錯誤: {str(e)}")
        return None, 0

def main():
    print("QuakeWatch - SAC 轉 TXT 格式")
    print("=" * 60)
    print(f"轉換倍數: {SCALE_FACTOR}")

    # 取得當前目錄
    current_dir = os.getcwd()
    print(f"目錄: {current_dir}\n")

    # 搜尋所有 SAC 檔案
    sac_files = glob.glob("*.SAC")

    if not sac_files:
        print("未找到任何 .SAC 檔案")
        return

    print(f"找到 {len(sac_files)} 個 SAC 檔案\n")

    # 轉換每個檔案
    success_count = 0
    total_points = 0

    for sac_file in sorted(sac_files):
        print(f"正在轉換: {sac_file}")
        txt_file, npts = convert_sac_to_txt(sac_file)

        if txt_file:
            print(f"✓ 已產生: {os.path.basename(txt_file)}")
            print(f"  資料點數: {npts}\n")
            success_count += 1
            total_points += npts

    print("=" * 60)
    print(f"轉換完成！")
    print(f"成功: {success_count}/{len(sac_files)} 個檔案")
    print(f"總資料點數: {total_points}")
    print("=" * 60)

if __name__ == "__main__":
    main()
