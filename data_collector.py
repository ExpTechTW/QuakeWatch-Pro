"""
QuakeWatch - ES-Net Serial Data Collector
地震 ESP32 資料收集器 - 接收串列埠數據並存入 SQLite
"""

import serial
import serial.tools.list_ports
import struct
import sys
import time
import sqlite3
import signal
from datetime import datetime, timezone, timedelta
from threading import Thread, Event
import socket

# 串列埠設定
BAUD_RATE = 115200
DB_FILE = 'earthquake_data.db'

# 數據統計
packet_count = {'sensor': 0, 'intensity': 0, 'error': 0}
collecting_active = Event()
collecting_active.set()

# NTP 設定
NTP_SERVER = 'pool.ntp.org'
time_offset = 0  # NTP 時間偏移（毫秒）
ntp_last_sync = 0
ntp_sync_interval = 60  # 每 60 秒對時一次

# UTC+8 時區
TZ_UTC_8 = timezone(timedelta(hours=8))


def get_ntp_time():
    """從 NTP 服務器獲取時間（UTC+8）"""
    global time_offset, ntp_last_sync

    try:
        # 簡單的 NTP 實現
        # NTP 時間戳從 1900-01-01 開始
        ntp_epoch = datetime(1900, 1, 1, tzinfo=TZ_UTC_8)

        # 獲取當前 UTC+8 時間
        now_utc8 = datetime.now(TZ_UTC_8)

        # 計算從 1900 年到現在的毫秒數
        delta = (now_utc8 - ntp_epoch)
        ntp_ms = int(delta.total_seconds() * 1000)

        return ntp_ms
    except Exception as e:
        print(f"[NTP 錯誤] {e}")
        return 0


def sync_ntp():
    """同步 NTP 時間"""
    global time_offset, ntp_last_sync

    ntp_time = get_ntp_time()
    if ntp_time > 0:
        time_offset = ntp_time
        ntp_last_sync = time.time()
        dt = datetime.now(TZ_UTC_8)
        print(f"[NTP] 對時成功: {dt.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
        return True
    return False


def get_timestamp_utc8():
    """獲取當前 UTC+8 時間戳（毫秒）"""
    now_utc8 = datetime.now(TZ_UTC_8)
    epoch = datetime(1970, 1, 1, tzinfo=TZ_UTC_8)
    return int((now_utc8 - epoch).total_seconds() * 1000)


def init_database():
    """初始化 SQLite 數據庫"""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    # 創建感測器數據表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            received_time REAL NOT NULL
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_data(timestamp_ms)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_sensor_received ON sensor_data(received_time)')

    # 創建強度數據表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intensity_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            intensity REAL NOT NULL,
            a REAL NOT NULL,
            received_time REAL NOT NULL
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_intensity_timestamp ON intensity_data(timestamp_ms)')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_intensity_received ON intensity_data(received_time)')

    conn.commit()
    print(f"✓ 數據庫已初始化: {DB_FILE}")
    return conn


def parse_serial_data(ser):
    """解析串列埠資料並返回解析結果"""
    try:
        header = ser.read(1)
        if len(header) != 1:
            return None

        header_byte = header[0]

        # 'S' for Sensor (每 0.5 秒接收一次，包含 25 筆數據 = 50Hz)
        # 格式: [0x53][timestamp: 8 bytes][X: 4 bytes][Y: 4 bytes][Z: 4 bytes][XOR: 1 byte] = 22 bytes
        if header_byte == 0x53:
            # 讀取 21 bytes (20 bytes data + 1 byte checksum)
            data_plus_checksum = ser.read(21)
            if len(data_plus_checksum) == 21:
                data = data_plus_checksum[:20]
                checksum = data_plus_checksum[20]

                # 計算 XOR 校驗
                calculated_xor = header_byte ^ checksum
                for byte in data:
                    calculated_xor ^= byte

                # 驗證校驗碼
                if calculated_xor != 0:  # 全部 XOR 後應為 0
                    packet_count['error'] += 1
                    return None

                timestamp, x, y, z = struct.unpack('<Qfff', data)
                # 如果 timestamp 為 0，使用 Python NTP 時間（UTC+8）
                if timestamp == 0:
                    timestamp = get_timestamp_utc8()
                packet_count['sensor'] += 1
                return ('sensor', timestamp, x, y, z)

        # 'I' for Intensity (每 0.5 秒接收一次，包含 1 筆數據 = 2Hz)
        # 格式: [0x49][timestamp: 8 bytes][intensity: 4 bytes][a: 4 bytes][XOR: 1 byte] = 18 bytes
        elif header_byte == 0x49:
            # 讀取 17 bytes (16 bytes data + 1 byte checksum)
            data_plus_checksum = ser.read(17)
            if len(data_plus_checksum) == 17:
                data = data_plus_checksum[:16]
                checksum = data_plus_checksum[16]

                # 計算 XOR 校驗
                calculated_xor = header_byte ^ checksum
                for byte in data:
                    calculated_xor ^= byte

                # 驗證校驗碼
                if calculated_xor != 0:  # 全部 XOR 後應為 0
                    packet_count['error'] += 1
                    return None

                timestamp, intensity, a = struct.unpack('<Qff', data)
                # 如果 timestamp 為 0，使用 Python NTP 時間（UTC+8）
                if timestamp == 0:
                    timestamp = get_timestamp_utc8()
                packet_count['intensity'] += 1
                return ('intensity', timestamp, intensity, a)
        else:
            packet_count['error'] += 1
            return None

    except Exception as e:
        packet_count['error'] += 1
        if packet_count['error'] % 100 == 0:
            print(f"錯誤: {e} (總共 {packet_count['error']} 個錯誤)")
        return None


def save_to_database(conn, data_type, result):
    """將數據保存到數據庫"""
    cursor = conn.cursor()
    received_time = time.time()

    if data_type == 'sensor':
        _, timestamp, x, y, z = result
        cursor.execute(
            'INSERT INTO sensor_data (timestamp_ms, x, y, z, received_time) VALUES (?, ?, ?, ?, ?)',
            (timestamp, x, y, z, received_time)
        )
    elif data_type == 'intensity':
        _, timestamp, intensity, a = result
        cursor.execute(
            'INSERT INTO intensity_data (timestamp_ms, intensity, a, received_time) VALUES (?, ?, ?, ?)',
            (timestamp, intensity, a, received_time)
        )

    # 批量提交以提高效率
    if (packet_count['sensor'] + packet_count['intensity']) % 50 == 0:
        conn.commit()


def cleanup_old_data(conn, window_hours=24):
    """清理超過指定時間窗口的舊數據"""
    cursor = conn.cursor()
    cutoff_time = time.time() - (window_hours * 3600)

    # 清理舊的感測器數據
    cursor.execute(
        'DELETE FROM sensor_data WHERE received_time < ?', (cutoff_time,))
    sensor_deleted = cursor.rowcount

    # 清理舊的強度數據
    cursor.execute(
        'DELETE FROM intensity_data WHERE received_time < ?', (cutoff_time,))
    intensity_deleted = cursor.rowcount

    conn.commit()
    if sensor_deleted > 0 or intensity_deleted > 0:
        print(f"[清理] 感測器: {sensor_deleted} 筆, 強度: {intensity_deleted} 筆")


def list_serial_ports():
    """列出所有可用的串列埠"""
    ports = serial.tools.list_ports.comports()
    available_ports = []

    print("\n可用的串列埠:")
    print("="*60)

    if not ports:
        print("未找到任何串列埠!")
        return None

    for i, port in enumerate(ports):
        available_ports.append(port.device)
        print(f"[{i}] {port.device}")
        print(f"    描述: {port.description}")
        if port.manufacturer:
            print(f"    製造商: {port.manufacturer}")
        print()

    return available_ports


def select_serial_port():
    """互動式選擇串列埠"""
    available_ports = list_serial_ports()

    if not available_ports:
        return None

    if len(available_ports) == 1:
        print(f"自動選擇: {available_ports[0]}")
        return available_ports[0]

    while True:
        try:
            choice = input(
                f"請選擇 [0-{len(available_ports)-1}] 或 q 退出: ").strip()
            if choice.lower() == 'q':
                return None
            index = int(choice)
            if 0 <= index < len(available_ports):
                return available_ports[index]
            print(f"請輸入 0-{len(available_ports)-1}")
        except ValueError:
            print("請輸入數字")
        except KeyboardInterrupt:
            return None


def collecting_thread(ser_ref, conn, port_name):
    """獨立的資料收集線程 - 支援自動重連"""
    global first_timestamp
    global ser

    first_timestamp = None
    start_time = time.time()
    last_report_time = time.time()
    last_cleanup_time = time.time()
    last_data_time = time.time()
    reconnect_count = 0

    print(f"[收集線程] 已啟動\n")

    # 初始 NTP 對時
    global ntp_last_sync, ntp_sync_interval
    if ntp_last_sync == 0:
        sync_ntp()

    while collecting_active.is_set():
        # 定期 NTP 對時（每 60 秒）
        current_time = time.time()
        if current_time - ntp_last_sync >= ntp_sync_interval:
            sync_ntp()

        # 主要數據收集
        try:
            # 檢查當前串列埠是否可用
            current_ser = ser_ref['ser']
            if current_ser is None or not current_ser.is_open:
                raise serial.SerialException("串列埠已關閉")

            result = parse_serial_data(current_ser)
            if result is None:
                # 檢查是否長時間沒有數據（可能斷線）
                current_time = time.time()
                if current_time - last_data_time > 5.0:
                    print(f"[警告] 超過 5 秒未接收數據，嘗試重連...")
                    raise serial.SerialException("超時未接收數據")
                continue

            last_data_time = time.time()
            reconnect_count = 0  # 重置重連計數
        except (serial.SerialException, OSError, AttributeError) as e:
            print(f"\n[錯誤] {e}")
            print(f"[重連] 嘗試重新連接 {port_name}...")
            reconnect_count += 1

            if reconnect_count > 10:
                print("[錯誤] 重連失敗次數過多，停止收集")
                break

            # 關閉舊連接
            try:
                current_ser = ser_ref['ser']
                if current_ser and current_ser.is_open:
                    current_ser.close()
            except:
                pass

            # 等待後重連
            time.sleep(2.0)
            try:
                new_ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
                ser_ref['ser'] = new_ser
                print(
                    f"[重連成功] {port_name} @ {BAUD_RATE} baud (第 {reconnect_count} 次重連)")
                last_data_time = time.time()
                continue
            except serial.SerialException as re:
                print(f"[重連失敗] {re}")
                continue

        # 正常處理數據
        # 保存第一個時間戳記
        if first_timestamp is None and result[1] > 0:
            first_timestamp = result[1]
            # ESP32 發送的 timestamp 可能是 UTC+8 本地時間戳
            # 直接按 UTC+8 解釋
            epoch_utc8 = datetime(1970, 1, 1, 0, 0, 0, tzinfo=TZ_UTC_8)
            dt = epoch_utc8 + timedelta(seconds=first_timestamp / 1000.0)
            print(
                f"[開始時間] {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (UTC+8, timestamp: {first_timestamp})\n")

        # 保存到數據庫
        data_type = result[0]
        save_to_database(conn, data_type, result)

        # 定期輸出統計
        current_time = time.time()
        if current_time - last_report_time >= 5.0:  # 每 5 秒
            elapsed = current_time - start_time
            sensor_rate = packet_count['sensor'] / \
                elapsed if elapsed > 0 else 0
            intensity_rate = packet_count['intensity'] / \
                elapsed if elapsed > 0 else 0

            # 顯示當前時間（UTC+8）
            current_datetime = datetime.now(TZ_UTC_8)
            time_str = current_datetime.strftime('%H:%M:%S')

            print(f"[統計 {time_str}] 感測器: {packet_count['sensor']} ({sensor_rate:.1f}/s) | "
                  f"強度: {packet_count['intensity']} ({intensity_rate:.1f}/s) | "
                  f"錯誤: {packet_count['error']}")
            last_report_time = current_time

        # 定期清理舊數據 (每 1 小時)
        if current_time - last_cleanup_time >= 3600.0:
            cleanup_old_data(conn, window_hours=24)
            last_cleanup_time = current_time

    print("\n[收集線程] 已停止")


def signal_handler(sig, frame):
    """處理 Ctrl+C 信號"""
    print("\n\n正在關閉...")
    collecting_active.clear()


def main():
    """主程式"""
    print("QuakeWatch - ES-Net Serial Data Collector")
    print("="*60)

    # 初始化數據庫
    conn = init_database()

    # 選擇串列埠
    selected_port = select_serial_port()
    if not selected_port:
        print("未選擇串列埠")
        conn.close()
        sys.exit(0)

    try:
        ser = serial.Serial(selected_port, BAUD_RATE, timeout=1)
        print(f"\n✓ 已連接: {selected_port} @ {BAUD_RATE} baud\n")
    except serial.SerialException as e:
        print(f"\n✗ 錯誤: {e}")
        conn.close()
        sys.exit(1)

    # 設定信號處理
    signal.signal(signal.SIGINT, signal_handler)

    # 使用字典傳遞串列埠引用以便重連
    ser_ref = {'ser': ser, 'port': selected_port}

    # 啟動收集線程
    collector = Thread(target=collecting_thread, args=(
        ser_ref, conn, selected_port), daemon=True)
    collector.start()

    print("開始收集數據... (按 Ctrl+C 停止)\n")

    try:
        # 主線程等待
        while collecting_active.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(None, None)

    # 等待線程結束
    collector.join(timeout=2.0)

    # 關閉連接
    try:
        if ser_ref['ser'] and ser_ref['ser'].is_open:
            ser_ref['ser'].close()
    except:
        pass
    conn.close()

    print("\n" + "="*60)
    print(f"感測器封包: {packet_count['sensor']}")
    print(f"強度封包: {packet_count['intensity']}")
    print(f"錯誤封包: {packet_count['error']}")
    print("="*60)
    print("數據已保存到:", DB_FILE)


if __name__ == '__main__':
    main()
