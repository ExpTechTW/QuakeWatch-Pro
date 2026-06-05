"""QuakeWatch 共用工具：中文字體、FFT/PSD 計算、資料庫讀取。"""

import sys
import sqlite3
import numpy as np


def setup_chinese_font():
    """依平台設定 matplotlib 中文字體。"""
    import matplotlib
    if sys.platform.startswith('win'):
        fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei']
    elif sys.platform == 'darwin':
        fonts = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'STHeiti']
    else:
        fonts = None
    if fonts:
        matplotlib.rcParams['font.sans-serif'] = fonts
        matplotlib.rcParams['axes.unicode_minus'] = False


# ===== FFT / PSD 參數（50 Hz 取樣、1024 點）=====
FFT_SIZE = 1024
FFT_FS = 50
FFT_FREQS_POS = np.fft.rfftfreq(FFT_SIZE, d=1.0 / FFT_FS)
FFT_WINDOW = np.hanning(FFT_SIZE).astype(np.float32)
_PSD_SCALE = 1.0 / (FFT_FS * FFT_SIZE)
_N_HALF = FFT_SIZE // 2 + 1


def compute_psd_db(fft_data):
    """由 FFT 結果計算單邊功率譜密度 (dB)，並裁切至 [-110, 0]。"""
    dft = fft_data[:_N_HALF]
    psd = _PSD_SCALE * np.abs(dft) ** 2
    psd[1:-1] *= 2  # 單邊化（除直流與 Nyquist 外加倍）
    return np.clip(10 * np.log10(psd + 1e-20), -110, 0)


def fetch_all(db_file, query, params=()):
    """開啟資料庫、執行查詢、回傳所有列，最後關閉連線。"""
    conn = sqlite3.connect(db_file)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()
