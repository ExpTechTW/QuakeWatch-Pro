"""
QuakeWatch - ES-Net Data Visualization
地震 ESP32 資料視覺化 - 從 SQLite 讀取數據並顯示圖表
"""

import sqlite3
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime, timezone, timedelta

# 數據庫文件
DB_FILE = 'earthquake_data.db'

# UTC+8 時區
TZ_UTC_8 = timezone(timedelta(hours=8))

# 資料窗參數 (基於時間)
DATA_WINDOW_LENGTH = 60  # 秒 - 保留最近 60 秒的資料

# 資料緩衝區設定 - 動態計算
# 接收頻率: 每 0.5 秒一次 (2 packets/sec)
# 感測器: 50Hz，強度: 2Hz
# 緩衝區 = DATA_WINDOW_LENGTH × 採樣率 × 1.2 (20% 餘裕)
MAX_SAMPLES_SENSOR = int(DATA_WINDOW_LENGTH * 50 *
                         1.2)      # 60 × 50 × 1.2 = 3600
MAX_SAMPLES_INTENSITY = int(
    DATA_WINDOW_LENGTH * 2 * 1.2)    # 60 × 2 × 1.2 = 144

x_data = deque(maxlen=MAX_SAMPLES_SENSOR)
y_data = deque(maxlen=MAX_SAMPLES_SENSOR)
z_data = deque(maxlen=MAX_SAMPLES_SENSOR)
time_data = deque(maxlen=MAX_SAMPLES_SENSOR)
timestamp_data = deque(maxlen=MAX_SAMPLES_SENSOR)  # NTP 時間戳記

# 濾波後的三軸資料
x_filtered = deque(maxlen=MAX_SAMPLES_SENSOR)
y_filtered = deque(maxlen=MAX_SAMPLES_SENSOR)
z_filtered = deque(maxlen=MAX_SAMPLES_SENSOR)

# 三向合成 PGA (Peak Ground Acceleration)
pga_raw = deque(maxlen=MAX_SAMPLES_SENSOR)       # 濾波前三向合成 (自己計算)
pga_filtered = deque(maxlen=MAX_SAMPLES_SENSOR)  # 濾波後三向合成 (自己計算)

intensity_history = deque(maxlen=MAX_SAMPLES_INTENSITY)
a_history = deque(maxlen=MAX_SAMPLES_INTENSITY)
intensity_time = deque(maxlen=MAX_SAMPLES_INTENSITY)
intensity_timestamp = deque(maxlen=MAX_SAMPLES_INTENSITY)  # NTP 時間戳記

packet_count = {'sensor': 0, 'intensity': 0, 'error': 0}
start_time = time.time()
first_timestamp = None  # 第一個收到的時間戳記
first_received_time = None  # 第一筆數據的 received_time

# 線程安全鎖
data_lock = threading.Lock()
parsing_active = threading.Event()
parsing_active.set()

# 解析統計
parse_stats = {
    'total_parsed': 0,
    'last_report_time': time.time(),
    'last_report_count': 0
}

# FFT 頻譜分析常數（預計算以提高效能）
FFT_SIZE = 2048
FFT_FS = 50  # 採樣率 50Hz
FFT_N = FFT_SIZE
FFT_FREQS_POS = np.fft.rfftfreq(FFT_N, d=1.0/FFT_FS)
FFT_WINDOW = np.hanning(FFT_SIZE)
FFT_PSD_SCALE = 1.0 / (FFT_FS * FFT_N)
N_HALF_PLUS_ONE = FFT_N // 2 + 1


def jma_low_cut_filter(f):
    """JMA Low-cut 濾波器: sqrt(1 - exp(-(f/0.5)^3))"""
    ratio = f / 0.5
    return np.sqrt(1.0 - np.exp(-np.power(ratio, 3.0)))


def jma_high_cut_filter(f):
    """JMA High-cut 濾波器"""
    y = f / 10.0
    y2 = y * y
    y4 = y2 * y2
    y6 = y4 * y2
    y8 = y6 * y2
    y10 = y8 * y2
    y12 = y10 * y2

    denominator = (1.0 + 0.694 * y2 + 0.241 * y4 + 0.0557 * y6 +
                   0.009664 * y8 + 0.00134 * y10 + 0.000155 * y12)

    return np.power(denominator, -0.5)


def jma_period_effect_filter(f):
    """JMA 周期效果濾波器: sqrt(1/f)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sqrt(1.0 / f)
        result[f == 0] = 0.0
    return result


def jma_combined_filter(f):
    """JMA 綜合濾波器"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = jma_low_cut_filter(
            f) * jma_high_cut_filter(f) * jma_period_effect_filter(f)
        result[f == 0] = 0.0
    return result


def compute_psd_db(fft_data):
    """計算功率譜密度並轉換為 dB"""
    # 只取正頻率部分
    dft = fft_data[:N_HALF_PLUS_ONE]

    # 計算 PSD
    psd = FFT_PSD_SCALE * np.abs(dft)**2

    # 對於非直流和奈奎斯特頻率，功率需要乘以 2
    psd[1:-1] *= 2

    # 轉換為 dB 並限制範圍
    psd_db = 10 * np.log10(psd + 1e-20)
    return np.clip(psd_db, -110, 0)


def apply_jma_filter(data, sampling_rate=50):
    """
    應用 JMA 綜合濾波器 (使用 FFT 頻域濾波)
    參考: ES-Net intensity.cpp 實現

    濾波步驟:
    1. FFT 轉換到頻域
    2. 應用 JMA 濾波器係數
    3. IFFT 轉回時域
    """
    if len(data) < 50:
        return data

    data_array = np.array(data, dtype=np.float32)  # 使用 float32 節省記憶體和運算時間
    n = len(data_array)

    # 補零到 2 的冪次方 (提高 FFT 效率)
    # 優化：使用位運算代替 log2
    n_pad = 1 << (n - 1).bit_length()  # 等同於 2 ** ceil(log2(n))，但更快

    # 1. 補零
    padded_data = np.pad(data_array, (0, n_pad - n), mode='constant')

    # 2. FFT 轉換
    fft_data = np.fft.fft(padded_data)

    # 3. 計算頻率軸（預計算 d 參數）
    freqs = np.fft.fftfreq(n_pad, d=1.0/sampling_rate)

    # 4. 計算 JMA 濾波器係數
    filter_coeffs = jma_combined_filter(np.abs(freqs))

    # 5. 在頻域應用濾波器
    filtered_fft = fft_data * filter_coeffs

    # 6. IFFT 轉回時域
    filtered_data = np.fft.ifft(filtered_fft).real

    # 7. 截取原始長度
    return filtered_data[:n]


def clean_old_data():
    """清理超過時間窗口的舊資料 (基於時間而非樣本數)"""
    # 使用 time_data 中的最大值作為當前時間
    if len(time_data) == 0:
        return

    # time_data 存儲的是相對於 first_received_time 的時間（以秒為單位）
    current_rel_time = time_data[-1]
    cutoff_time = current_rel_time - DATA_WINDOW_LENGTH

    # 清理感測器資料 - 使用索引批量刪除
    if len(time_data) > 0:
        # 找到需要保留的第一個索引
        remove_count = 0
        for t in time_data:
            if t >= cutoff_time:
                break
            remove_count += 1

        # 批量刪除舊數據
        if remove_count > 0:
            for _ in range(remove_count):
                time_data.popleft()
                x_data.popleft()
                y_data.popleft()
                z_data.popleft()
                timestamp_data.popleft()

    # 清理強度資料
    if len(intensity_time) > 0:
        remove_count = 0
        for t in intensity_time:
            if t >= cutoff_time:
                break
            remove_count += 1

        if remove_count > 0:
            for _ in range(remove_count):
                intensity_time.popleft()
                intensity_history.popleft()
                a_history.popleft()
                intensity_timestamp.popleft()


def parsing_thread():
    """獨立的資料解析線程 - 從 SQLite 讀取數據"""
    global first_timestamp
    global first_received_time

    first_received_time = None

    print(f"[解析線程] 已啟動 (時間窗口: {DATA_WINDOW_LENGTH} 秒)\n")
    report_interval = 1.0
    clean_interval = 2.0
    filter_interval = 0.5
    last_clean_time = time.time()
    last_filter_time = time.time()
    last_received_time_sensor = None
    last_received_time_intensity = None

    while parsing_active.is_set():
        # 從 SQLite 讀取新數據
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        cursor = conn.cursor()

        try:
            # 獲取感測器數據（往前讀取 60 秒）
            cutoff_time = time.time() - 60

            if last_received_time_sensor is None:
                # 首次運行，讀取最近 60 秒的數據
                cursor.execute(
                    'SELECT timestamp_ms, x, y, z, received_time FROM sensor_data WHERE received_time > ? ORDER BY received_time ASC',
                    (cutoff_time,))
                sensor_rows = cursor.fetchall()
                if sensor_rows:
                    last_received_time_sensor = sensor_rows[-1][4]
            else:
                # 繼續讀取新數據，但從 60 秒前開始
                cursor.execute(
                    'SELECT timestamp_ms, x, y, z, received_time FROM sensor_data WHERE received_time > ? AND received_time > ? ORDER BY received_time ASC',
                    (max(last_received_time_sensor, cutoff_time), cutoff_time))
                sensor_rows = cursor.fetchall()

            if sensor_rows:
                with data_lock:
                    for row in sensor_rows:
                        timestamp, x, y, z, received_time = row
                        last_received_time_sensor = received_time
                        # 設定第一個時間點
                        if first_received_time is None:
                            first_received_time = received_time
                        if first_timestamp is None and timestamp > 0:
                            first_timestamp = timestamp

                        x_data.append(x)
                        y_data.append(y)
                        z_data.append(z)
                        # 使用 received_time 來計算相對時間，保持原始採樣率
                        time_data.append(received_time - first_received_time)
                        timestamp_data.append(timestamp)
                        parse_stats['total_parsed'] += 1

            # 獲取強度數據
            if last_received_time_intensity is None:
                # 首次運行，讀取最近 60 秒的數據
                cursor.execute(
                    'SELECT timestamp_ms, intensity, a, received_time FROM intensity_data WHERE received_time > ? ORDER BY received_time ASC',
                    (cutoff_time,))
                intensity_rows = cursor.fetchall()
                if intensity_rows:
                    last_received_time_intensity = intensity_rows[-1][3]
            else:
                # 繼續讀取新數據
                cursor.execute(
                    'SELECT timestamp_ms, intensity, a, received_time FROM intensity_data WHERE received_time > ? AND received_time > ? ORDER BY received_time ASC',
                    (max(last_received_time_intensity, cutoff_time), cutoff_time))
                intensity_rows = cursor.fetchall()

            if intensity_rows:
                with data_lock:
                    for row in intensity_rows:
                        timestamp, intensity, a, received_time = row
                        last_received_time_intensity = received_time
                        if first_timestamp is None and timestamp > 0:
                            first_timestamp = timestamp
                        intensity_history.append(intensity)
                        a_history.append(a)
                        intensity_time.append(
                            received_time - first_received_time)
                        intensity_timestamp.append(timestamp)
                        parse_stats['total_parsed'] += 1

            conn.close()

        except Exception as e:
            if conn:
                conn.close()

        # 添加延遲避免過度查詢數據庫
        time.sleep(0.1)

        # 定期清理超過時間窗口的舊資料
        current_check_time = time.time()
        if current_check_time - last_clean_time >= clean_interval:
            with data_lock:
                clean_old_data()
            last_clean_time = current_check_time

        # 定期應用 JMA 濾波器（每 0.5 秒更新一次，避免過於頻繁）
        if current_check_time - last_filter_time >= filter_interval:
            if len(x_data) >= 50:  # 至少 1 秒的數據 (50Hz)
                with data_lock:
                    try:
                        # 1. 計算濾波前三向合成 PGA（一次性轉換為陣列）
                        # 使用 float32 節省記憶體
                        x_arr = np.array(x_data, dtype=np.float32)
                        y_arr = np.array(y_data, dtype=np.float32)
                        z_arr = np.array(z_data, dtype=np.float32)
                        pga_raw_arr = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)

                        # 直接使用 extend 批量更新 deque
                        pga_raw.clear()
                        pga_raw.extend(pga_raw_arr)

                        # 2. 應用 JMA 濾波器到三軸
                        x_filt = apply_jma_filter(list(x_data))
                        y_filt = apply_jma_filter(list(y_data))
                        z_filt = apply_jma_filter(list(z_data))

                        # 確保濾波結果長度與原始數據相同
                        if len(x_filt) == len(x_data):
                            # 更新濾波後的緩衝區
                            x_filtered.clear()
                            x_filtered.extend(x_filt)
                            y_filtered.clear()
                            y_filtered.extend(y_filt)
                            z_filtered.clear()
                            z_filtered.extend(z_filt)

                            # 3. 計算濾波後三向合成 PGA
                            pga_filt_arr = np.sqrt(
                                x_filt**2 + y_filt**2 + z_filt**2)
                            pga_filtered.clear()
                            pga_filtered.extend(pga_filt_arr)
                    except Exception as e:
                        pass  # 濾波失敗時靜默跳過

            last_filter_time = current_check_time

        # 定期輸出解析統計
        if current_check_time - parse_stats['last_report_time'] >= report_interval:
            with data_lock:
                parsed_count = parse_stats['total_parsed'] - \
                    parse_stats['last_report_count']
                rate = parsed_count / report_interval
                # 計算實際時間跨度
                time_span = time_data[-1] - \
                    time_data[0] if len(time_data) > 1 else 0

    print("[解析線程] 已停止")


# 全局變量：追蹤上一次更新時間，用於控制 FFT 計算頻率
_last_fft_update_time = 0
_fft_update_interval = 0.3  # 每 300ms 更新一次 FFT，而不是每 50ms


def update_plot(frame):
    """更新圖表 - 僅從緩衝區讀取資料"""
    global _last_fft_update_time

    current_time = time.time() - start_time
    should_update_fft = (
        current_time - _last_fft_update_time) >= _fft_update_interval

    with data_lock:
        # 計算 X 軸範圍 - 使用 time_data 的相對時間
        if len(time_data) > 0:
            current_rel_time = time_data[-1]
            x_min = max(0, current_rel_time - DATA_WINDOW_LENGTH)
            x_max = current_rel_time
        else:
            x_min = 0
            x_max = DATA_WINDOW_LENGTH

        # 提前轉換為 list，避免重複轉換
        data_len = len(time_data)
        if data_len > 0:
            time_list = list(time_data)
            x_list = list(x_data)
            y_list = list(y_data)
            z_list = list(z_data)
        else:
            time_list = x_list = y_list = z_list = []

        # 圖表1: 三軸加速度
        if data_len > 0:
            line_x.set_data(time_list, x_list)
            line_y.set_data(time_list, y_list)
            line_z.set_data(time_list, z_list)
            ax1.set_xlim(x_min, x_max)

        # 圖表2: 三軸濾波（提前轉換以便後續重用）
        x_filt_list = []
        y_filt_list = []
        z_filt_list = []
        if len(x_filtered) == data_len:
            # 只在需要時轉換
            x_filt_list = list(x_filtered)
            y_filt_list = list(y_filtered)
            z_filt_list = list(z_filtered)
            line_x_filt.set_data(time_list, x_filt_list)
            line_y_filt.set_data(time_list, y_filt_list)
            line_z_filt.set_data(time_list, z_filt_list)
        ax2.set_xlim(x_min, x_max)

        # 圖表3, 圖表4: 三軸頻譜分析
        # 只在需要時更新 FFT（降低計算頻率，減少延遲）
        if should_update_fft and data_len >= FFT_SIZE:  # 足夠的數據進行 FFT
            # 重用前面轉換的 list，只取最新的 FFT_SIZE 個樣本
            x_arr = np.array(x_list[-FFT_SIZE:], dtype=np.float32)
            y_arr = np.array(y_list[-FFT_SIZE:], dtype=np.float32)
            z_arr = np.array(z_list[-FFT_SIZE:], dtype=np.float32)

            # 應用 Hanning 窗函數並計算 FFT
            fft_x = np.fft.fft(x_arr * FFT_WINDOW)
            fft_y = np.fft.fft(y_arr * FFT_WINDOW)
            fft_z = np.fft.fft(z_arr * FFT_WINDOW)

            # 計算 PSD 並轉換為 dB
            line_fft_x.set_data(FFT_FREQS_POS, compute_psd_db(fft_x))
            line_fft_y.set_data(FFT_FREQS_POS, compute_psd_db(fft_y))
            line_fft_z.set_data(FFT_FREQS_POS, compute_psd_db(fft_z))

            # 濾波後三軸頻譜
            if len(x_filtered) >= FFT_SIZE and len(x_filt_list) > 0:
                # 重用前面轉換的 filtered list
                x_filt_arr = np.array(
                    x_filt_list[-FFT_SIZE:], dtype=np.float32)
                y_filt_arr = np.array(
                    y_filt_list[-FFT_SIZE:], dtype=np.float32)
                z_filt_arr = np.array(
                    z_filt_list[-FFT_SIZE:], dtype=np.float32)

                # 應用 Hanning 窗函數並計算 FFT
                fft_x_filt = np.fft.fft(x_filt_arr * FFT_WINDOW)
                fft_y_filt = np.fft.fft(y_filt_arr * FFT_WINDOW)
                fft_z_filt = np.fft.fft(z_filt_arr * FFT_WINDOW)

                # 計算 PSD 並轉換為 dB
                line_fft_x_filt.set_data(
                    FFT_FREQS_POS, compute_psd_db(fft_x_filt))
                line_fft_y_filt.set_data(
                    FFT_FREQS_POS, compute_psd_db(fft_y_filt))
                line_fft_z_filt.set_data(
                    FFT_FREQS_POS, compute_psd_db(fft_z_filt))

            # 更新 FFT 計算時間戳（在鎖內更新）
            _last_fft_update_time = current_time

        # 圖表5: PGA(未濾波) + PGA(濾波a) + 震度
        if data_len > 0 and len(pga_raw) == data_len:
            # 重用已轉換的 list
            pga_raw_list = list(pga_raw)
            line_pga_raw_5.set_data(time_list, pga_raw_list)

        if len(intensity_time) > 0:
            # 轉換強度相關數據
            intensity_time_list = list(intensity_time)
            a_history_list = list(a_history)
            intensity_history_list = list(intensity_history)
            line_pga_filt_5.set_data(intensity_time_list, a_history_list)
            line_i.set_data(intensity_time_list, intensity_history_list)

        # 更新 ax5 X 軸
        ax5.set_xlim(x_min, x_max)

    # 手動觸發所有圖表重繪（但不要每次都 flush_events，這樣可以減少延遲）
    for fig in [fig1, fig2, fig3, fig4, fig5]:
        fig.canvas.draw_idle()
    # 只在最後一個圖表時才 flush_events
    fig5.canvas.flush_events()


def print_statistics():
    """顯示統計"""
    from datetime import datetime

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"執行時間: {elapsed:.1f} 秒")

    if elapsed > 0:
        print(
            f"感測器封包: {packet_count['sensor']} ({packet_count['sensor']/elapsed:.1f} Hz)")
        print(
            f"強度封包: {packet_count['intensity']} ({packet_count['intensity']/elapsed:.1f} Hz)")
        print(f"錯誤封包: {packet_count['error']}")

    # 顯示時間戳記資訊（UTC）
    if first_timestamp is not None and first_timestamp > 0:
        dt = datetime.fromtimestamp(first_timestamp / 1000.0, tz=timezone.utc)
        print(f"\n首次時間戳記: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")

        if len(timestamp_data) > 0:
            latest = timestamp_data[-1]
            if latest > 0:
                dt_latest = datetime.fromtimestamp(
                    latest / 1000.0, tz=timezone.utc)
                print(
                    f"最新時間戳記: {dt_latest.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")

                # 計算時間跨度
                duration_ms = latest - first_timestamp
                print(f"資料時間跨度: {duration_ms / 1000.0:.2f} 秒")
    else:
        print("\n⚠ 未接收到有效的 NTP 時間戳記")

    print("="*60)


def main():
    """主程式"""
    global ax1, ax2, ax3, ax4, ax5, fig1, fig2, fig3, fig4, fig5
    global line_x, line_y, line_z
    global line_x_filt, line_y_filt, line_z_filt
    global line_fft_x, line_fft_y, line_fft_z
    global line_fft_x_filt, line_fft_y_filt, line_fft_z_filt
    global line_pga_raw_5, line_pga_filt_5, line_i

    print("QuakeWatch - ES-Net Data Visualization")
    print("="*60)

    # 檢查數據庫文件是否存在
    import os
    if not os.path.exists(DB_FILE):
        print(f"\n✗ 錯誤: 找不到數據庫文件 {DB_FILE}")
        print("請先運行 python3 data_collector.py 收集數據")
        sys.exit(1)

    print(f"\n✓ 數據庫文件: {DB_FILE}")

    # 設定圖表字體
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',
                                           'Heiti TC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 使用深色主題
    plt.style.use('dark_background')

    # 創建5個獨立窗口
    fig1 = plt.figure(num='圖表1: 三軸加速度', figsize=(10, 5))
    fig2 = plt.figure(num='圖表2: 三軸濾波', figsize=(10, 5))
    fig3 = plt.figure(num='圖表3: 三軸頻譜', figsize=(10, 5))
    fig4 = plt.figure(num='圖表4: 三軸濾波頻譜', figsize=(10, 5))
    fig5 = plt.figure(num='圖表5: PGA + 震度', figsize=(12, 5))

    for fig in [fig1, fig2, fig3, fig4, fig5]:
        fig.patch.set_facecolor('#0d1117')

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)
    ax4 = fig4.add_subplot(111)
    ax5 = fig5.add_subplot(111)

    # === 圖表1: 三軸加速度 ===
    ax1.set_facecolor('#161b22')
    ax1.set_title('三軸加速度',
                  fontsize=14, fontweight='bold', color='#58a6ff', pad=12)
    ax1.set_xlabel('時間 (秒)', fontsize=11)
    ax1.set_ylabel('加速度 (Gal)', fontsize=11)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)

    line_x, = ax1.plot([], [], '#ff6b6b', label='X 軸',
                       linewidth=1.3, alpha=0.85)
    line_y, = ax1.plot([], [], '#4ecdc4', label='Y 軸',
                       linewidth=1.3, alpha=0.85)
    line_z, = ax1.plot([], [], '#45b7d1', label='Z 軸',
                       linewidth=1.3, alpha=0.85)

    ax1.legend(loc='upper right', fontsize=10, framealpha=0.8)
    ax1.set_ylim(-5, 5)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.7, alpha=0.3)
    fig1.tight_layout()

    # === 圖表2: 三軸濾波後加速度 ===
    ax2.set_facecolor('#161b22')
    ax2.set_title('三軸濾波',
                  fontsize=14, fontweight='bold', color='#58a6ff', pad=12)
    ax2.set_xlabel('時間 (秒)', fontsize=11)
    ax2.set_ylabel('加速度 (Gal)', fontsize=11)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)

    line_x_filt, = ax2.plot(
        [], [], '#ff6b6b', label='X 軸', linewidth=1.3, alpha=0.85)
    line_y_filt, = ax2.plot(
        [], [], '#4ecdc4', label='Y 軸', linewidth=1.3, alpha=0.85)
    line_z_filt, = ax2.plot(
        [], [], '#45b7d1', label='Z 軸', linewidth=1.3, alpha=0.85)

    ax2.legend(loc='upper right', fontsize=10, framealpha=0.8)
    ax2.set_ylim(-5, 5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.7, alpha=0.3)
    fig2.tight_layout()

    # === 圖表3: 三軸頻譜分析 (未濾波) ===
    ax3.set_facecolor('#161b22')
    ax3.set_title('三軸頻譜 (未濾波, 0-25Hz)',
                  fontsize=13, fontweight='bold', color='#58a6ff', pad=10)
    ax3.set_xlabel('頻率 (Hz)', fontsize=10)
    ax3.set_ylabel('功率譜密度 (dB)', fontsize=10)
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.6, which='both')
    ax3.set_xlim(0, 25)  # Nyquist 頻率 = 取樣率/2
    ax3.set_ylim(-110, 0)

    line_fft_x, = ax3.plot([], [], '#ff6b6b', label='X 軸',
                           linewidth=1.2, alpha=0.8)
    line_fft_y, = ax3.plot([], [], '#4ecdc4', label='Y 軸',
                           linewidth=1.2, alpha=0.8)
    line_fft_z, = ax3.plot([], [], '#45b7d1', label='Z 軸',
                           linewidth=1.2, alpha=0.8)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.7)
    print(
        f"[INIT] 圖表3 線條: X={id(line_fft_x)}, Y={id(line_fft_y)}, Z={id(line_fft_z)}")
    print(f"[INIT] 圖表3 總線條數: {len(ax3.get_lines())}")

    # === 圖表4: 三軸頻譜分析 (濾波後) ===
    ax4.set_facecolor('#161b22')
    ax4.set_title('三軸頻譜 (濾波後, 0-25Hz)',
                  fontsize=13, fontweight='bold', color='#58a6ff', pad=10)
    ax4.set_xlabel('頻率 (Hz)', fontsize=10)
    ax4.set_ylabel('功率譜密度 (dB)', fontsize=10)
    ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.6, which='both')
    ax4.set_xlim(0, 25)  # Nyquist 頻率 = 取樣率/2
    ax4.set_ylim(-110, 0)

    line_fft_x_filt, = ax4.plot(
        [], [], '#ff6b6b', label='X 軸', linewidth=1.2, alpha=0.8)
    line_fft_y_filt, = ax4.plot(
        [], [], '#4ecdc4', label='Y 軸', linewidth=1.2, alpha=0.8)
    line_fft_z_filt, = ax4.plot(
        [], [], '#45b7d1', label='Z 軸', linewidth=1.2, alpha=0.8)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.7)
    print(
        f"[INIT] 圖表4 線條: X={id(line_fft_x_filt)}, Y={id(line_fft_y_filt)}, Z={id(line_fft_z_filt)}")
    print(f"[INIT] 圖表4 總線條數: {len(ax4.get_lines())}")

    # === 圖表5: PGA(未濾波) + PGA(濾波) + 計測震度 + 震度階級 ===
    ax5.set_facecolor('#161b22')
    ax5_twin = ax5.twinx()  # 右側Y軸：震度
    ax5_twin.set_facecolor('#161b22')

    ax5.set_title('PGA + 計測震度 + 震度階級',
                  fontsize=13, fontweight='bold', color='#58a6ff', pad=10)
    ax5.set_xlabel('時間 (秒)', fontsize=10)
    ax5.set_ylabel('PGA (Gal)', fontsize=10, color='white')
    ax5_twin.set_ylabel('震度', fontsize=10, color='#ffd93d')
    ax5.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)

    # 左軸：PGA
    line_pga_raw_5, = ax5.plot([], [], '#ff9500', label='PGA 未濾波',
                               linewidth=1.8, alpha=0.85)
    line_pga_filt_5, = ax5.plot([], [], '#6bcf7f', label='PGA 濾波 (a)',
                                linewidth=2, alpha=0.95)

    # 右軸：震度
    line_i, = ax5_twin.plot([], [], '#ffd93d', label='計測震度',
                            linewidth=2.5, marker='o', markersize=5,
                            markerfacecolor='#ffd93d', markeredgecolor='white',
                            markeredgewidth=0.6, alpha=0.95)

    ax5.set_ylim(-1, 30)
    ax5_twin.set_ylim(-0.5, 7)
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=0.6, alpha=0.3)

    # 震度階級參考線
    for level in [1, 2, 3, 4, 5]:
        ax5_twin.axhline(y=level, color='gray', linestyle=':',
                         linewidth=0.5, alpha=0.25)

    # 合併圖例
    lines_leg = [line_pga_raw_5, line_pga_filt_5, line_i]
    labels_leg = ['PGA 未濾波', 'PGA 濾波 (a)', '計測震度']
    ax5.legend(lines_leg, labels_leg, loc='upper right',
               fontsize=10, framealpha=0.8)
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()

    print("\n開始接收資料...\n")

    # 啟動獨立的解析線程（從 SQLite 讀取）
    parser = threading.Thread(target=parsing_thread, args=(), daemon=True)
    parser.start()

    # 圖表更新動畫 - 只負責顯示緩衝區資料
    # 使用 fig1 作為主動畫對象
    # blit=False 避免頻譜圖表出現錯誤連線
    ani = FuncAnimation(fig1, update_plot, interval=50,
                        blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n程式終止")
    finally:
        # 停止解析線程
        parsing_active.clear()
        parser.join(timeout=2.0)

        print_statistics()


if __name__ == '__main__':
    main()
