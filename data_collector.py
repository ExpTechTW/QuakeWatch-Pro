"""
QuakeWatch - ES-Net Serial Data Collector
地震 ESP32 資料收集器
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

BAUD_RATE = 115200
DB_FILE = 'earthquake_data.db'

packet_count = {'sensor': 0, 'intensity': 0, 'filtered': 0, 'error': 0}
collecting_active = Event()
collecting_active.set()

time_offset = 0
ntp_last_sync = 0
ntp_sync_interval = 60
TZ_UTC_8 = timezone(timedelta(hours=8))


def get_timestamp_utc8():
    now_utc8 = datetime.now(TZ_UTC_8)
    epoch = datetime(1970, 1, 1, tzinfo=TZ_UTC_8)
    return int((now_utc8 - epoch).total_seconds() * 1000)


def sync_ntp():
    global ntp_last_sync
    ntp_last_sync = time.time()
    dt = datetime.now(TZ_UTC_8)
    print(f"[NTP] 對時成功: {dt.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")


def init_database():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

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
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_data(timestamp_ms)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_received ON sensor_data(received_time)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intensity_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            intensity REAL NOT NULL,
            a REAL NOT NULL,
            received_time REAL NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_intensity_timestamp ON intensity_data(timestamp_ms)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_intensity_received ON intensity_data(received_time)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS filtered_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            h1 REAL NOT NULL,
            h2 REAL NOT NULL,
            v REAL NOT NULL,
            received_time REAL NOT NULL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_timestamp ON filtered_data(timestamp_ms)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_filtered_received ON filtered_data(received_time)')

    conn.commit()
    print(f"✓ 數據庫已初始化: {DB_FILE}")
    return conn


def parse_serial_data(ser):
    try:
        header = ser.read(1)
        if len(header) != 1:
            return None

        header_byte = header[0]

        if header_byte == 0x53:
            data_plus_checksum = ser.read(21)
            if len(data_plus_checksum) == 21:
                data = data_plus_checksum[:20]
                checksum = data_plus_checksum[20]

                calculated_xor = header_byte ^ checksum
                for byte in data:
                    calculated_xor ^= byte

                if calculated_xor != 0:
                    packet_count['error'] += 1
                    return None

                timestamp, x, y, z = struct.unpack('<Qfff', data)
                packet_count['sensor'] += 1
                return ('sensor', timestamp, x, y, z)

        elif header_byte == 0x49:
            data_plus_checksum = ser.read(17)
            if len(data_plus_checksum) == 17:
                data = data_plus_checksum[:16]
                checksum = data_plus_checksum[16]

                calculated_xor = header_byte ^ checksum
                for byte in data:
                    calculated_xor ^= byte

                if calculated_xor != 0:
                    packet_count['error'] += 1
                    return None

                timestamp, intensity, a = struct.unpack('<Qff', data)
                if timestamp == 0:
                    timestamp = get_timestamp_utc8()
                packet_count['intensity'] += 1
                return ('intensity', timestamp, intensity, a)

        elif header_byte == 0x46:
            data_plus_checksum = ser.read(21)
            if len(data_plus_checksum) == 21:
                data = data_plus_checksum[:20]
                checksum = data_plus_checksum[20]

                calculated_xor = header_byte ^ checksum
                for byte in data:
                    calculated_xor ^= byte

                if calculated_xor != 0:
                    packet_count['error'] += 1
                    return None

                timestamp, h1, h2, v = struct.unpack('<Qfff', data)
                packet_count['filtered'] += 1
                return ('filtered', timestamp, h1, h2, v)
        else:
            packet_count['error'] += 1
            return None

    except Exception as e:
        packet_count['error'] += 1
        if packet_count['error'] % 100 == 0:
            print(f"錯誤: {e} (總共 {packet_count['error']} 個錯誤)")
        return None


def process_batch(conn, data_type, batch, last_batch_time, current_batch_time):
    if len(batch) == 0:
        return

    cursor = conn.cursor()
    received_time_base = time.time()
    samples_count = len(batch)
    first_timestamp = batch[0][1]

    if first_timestamp > 0:
        for result in batch:
            if data_type == 'sensor':
                _, timestamp, x, y, z = result
                cursor.execute(
                    'INSERT INTO sensor_data (timestamp_ms, x, y, z, received_time) VALUES (?, ?, ?, ?, ?)',
                    (timestamp, x, y, z, received_time_base)
                )
            elif data_type == 'filtered':
                _, timestamp, h1, h2, v = result
                cursor.execute(
                    'INSERT INTO filtered_data (timestamp_ms, h1, h2, v, received_time) VALUES (?, ?, ?, ?, ?)',
                    (timestamp, h1, h2, v, received_time_base)
                )
    else:
        interval_ms = current_batch_time - last_batch_time
        time_step = interval_ms / samples_count

        for index, result in enumerate(batch):
            adjusted_timestamp = int(last_batch_time + (index + 1) * time_step)

            if data_type == 'sensor':
                _, timestamp, x, y, z = result
                cursor.execute(
                    'INSERT INTO sensor_data (timestamp_ms, x, y, z, received_time) VALUES (?, ?, ?, ?, ?)',
                    (adjusted_timestamp, x, y, z, received_time_base)
                )
            elif data_type == 'filtered':
                _, timestamp, h1, h2, v = result
                cursor.execute(
                    'INSERT INTO filtered_data (timestamp_ms, h1, h2, v, received_time) VALUES (?, ?, ?, ?, ?)',
                    (adjusted_timestamp, h1, h2, v, received_time_base)
                )

    conn.commit()


def save_to_database(conn, data_type, result):
    cursor = conn.cursor()
    received_time = time.time()

    if data_type == 'intensity':
        _, timestamp, intensity, a = result
        cursor.execute(
            'INSERT INTO intensity_data (timestamp_ms, intensity, a, received_time) VALUES (?, ?, ?, ?)',
            (timestamp, intensity, a, received_time)
        )
        conn.commit()


def cleanup_old_data(conn, window_hours=24):
    cursor = conn.cursor()
    cutoff_time = time.time() - (window_hours * 3600)

    cursor.execute('DELETE FROM sensor_data WHERE received_time < ?', (cutoff_time,))
    sensor_deleted = cursor.rowcount

    cursor.execute('DELETE FROM intensity_data WHERE received_time < ?', (cutoff_time,))
    intensity_deleted = cursor.rowcount

    cursor.execute('DELETE FROM filtered_data WHERE received_time < ?', (cutoff_time,))
    filtered_deleted = cursor.rowcount

    conn.commit()
    if sensor_deleted > 0 or intensity_deleted > 0 or filtered_deleted > 0:
        print(f"[清理] 感測器: {sensor_deleted} 筆, 強度: {intensity_deleted} 筆, 過濾: {filtered_deleted} 筆")


def list_serial_ports():
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
    available_ports = list_serial_ports()

    if not available_ports:
        return None

    if len(available_ports) == 1:
        print(f"自動選擇: {available_ports[0]}")
        return available_ports[0]

    while True:
        try:
            choice = input(f"請選擇 [0-{len(available_ports)-1}] 或 q 退出: ").strip()
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
    global first_timestamp
    global ser

    first_timestamp = None
    start_time = time.time()
    last_report_time = time.time()
    last_cleanup_time = time.time()
    last_data_time = time.time()
    reconnect_count = 0

    batch_buffer = {'sensor': [], 'filtered': []}
    last_receive_time = {'sensor': 0, 'filtered': 0}
    last_batch_timestamp = {'sensor': 0, 'filtered': 0}
    BATCH_TIMEOUT = 0.1

    print(f"[收集線程] 已啟動\n")

    global ntp_last_sync, ntp_sync_interval
    if ntp_last_sync == 0:
        sync_ntp()

    while collecting_active.is_set():
        current_time = time.time()
        if current_time - ntp_last_sync >= ntp_sync_interval:
            sync_ntp()

        try:
            current_ser = ser_ref['ser']
            if current_ser is None or not current_ser.is_open:
                raise serial.SerialException("串列埠已關閉")

            result = parse_serial_data(current_ser)
            if result is None:
                current_time = time.time()
                if current_time - last_data_time > 5.0:
                    print(f"[警告] 超過 5 秒未接收數據，嘗試重連...")
                    raise serial.SerialException("超時未接收數據")
                continue

            last_data_time = time.time()
            reconnect_count = 0
        except (serial.SerialException, OSError, AttributeError) as e:
            print(f"\n[錯誤] {e}")
            print(f"[重連] 嘗試重新連接 {port_name}...")
            reconnect_count += 1

            if reconnect_count > 10:
                print("[錯誤] 重連失敗次數過多，停止收集")
                break

            try:
                current_ser = ser_ref['ser']
                if current_ser and current_ser.is_open:
                    current_ser.close()
            except:
                pass

            time.sleep(2.0)
            try:
                new_ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
                ser_ref['ser'] = new_ser
                print(f"[重連成功] {port_name} @ {BAUD_RATE} baud (第 {reconnect_count} 次重連)")
                last_data_time = time.time()
                continue
            except serial.SerialException as re:
                print(f"[重連失敗] {re}")
                continue

        if first_timestamp is None and result[1] > 0:
            first_timestamp = result[1]
            epoch_utc8 = datetime(1970, 1, 1, 0, 0, 0, tzinfo=TZ_UTC_8)
            dt = epoch_utc8 + timedelta(seconds=first_timestamp / 1000.0)
            print(f"[開始時間] {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (UTC+8, timestamp: {first_timestamp})\n")

        data_type = result[0]
        current_receive_time = time.time()
        current_timestamp_ms = get_timestamp_utc8()

        if data_type in ['sensor', 'filtered']:
            time_since_last = current_receive_time - last_receive_time[data_type]

            if time_since_last < BATCH_TIMEOUT and len(batch_buffer[data_type]) > 0:
                batch_buffer[data_type].append(result)
            else:
                if len(batch_buffer[data_type]) > 0:
                    prev_timestamp = last_batch_timestamp[data_type]
                    if prev_timestamp == 0:
                        prev_timestamp = current_timestamp_ms - 500

                    process_batch(conn, data_type, batch_buffer[data_type],
                                prev_timestamp, current_timestamp_ms)
                    batch_buffer[data_type] = []
                    last_batch_timestamp[data_type] = current_timestamp_ms

                batch_buffer[data_type].append(result)
                last_receive_time[data_type] = current_receive_time
        else:
            save_to_database(conn, data_type, result)

        current_time = time.time()
        if current_time - last_report_time >= 5.0:
            elapsed = current_time - start_time
            sensor_rate = packet_count['sensor'] / elapsed if elapsed > 0 else 0
            intensity_rate = packet_count['intensity'] / elapsed if elapsed > 0 else 0
            filtered_rate = packet_count['filtered'] / elapsed if elapsed > 0 else 0

            current_datetime = datetime.now(TZ_UTC_8)
            time_str = current_datetime.strftime('%H:%M:%S')

            print(f"[統計 {time_str}] 感測器: {packet_count['sensor']} ({sensor_rate:.1f}/s) | "
                  f"強度: {packet_count['intensity']} ({intensity_rate:.1f}/s) | "
                  f"過濾: {packet_count['filtered']} ({filtered_rate:.1f}/s) | "
                  f"錯誤: {packet_count['error']}")
            last_report_time = current_time

        if current_time - last_cleanup_time >= 3600.0:
            cleanup_old_data(conn, window_hours=24)
            last_cleanup_time = current_time

    print("\n[收集線程] 已停止")


def signal_handler(sig, frame):
    print("\n\n正在關閉...")
    collecting_active.clear()


def main():
    print("QuakeWatch - ES-Net Serial Data Collector")
    print("="*60)

    conn = init_database()

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

    signal.signal(signal.SIGINT, signal_handler)

    ser_ref = {'ser': ser, 'port': selected_port}

    collector = Thread(target=collecting_thread, args=(ser_ref, conn, selected_port), daemon=True)
    collector.start()

    print("開始收集數據... (按 Ctrl+C 停止)\n")

    try:
        while collecting_active.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        signal_handler(None, None)

    collector.join(timeout=2.0)

    try:
        if ser_ref['ser'] and ser_ref['ser'].is_open:
            ser_ref['ser'].close()
    except:
        pass
    conn.close()

    print("\n" + "="*60)
    print(f"感測器封包: {packet_count['sensor']}")
    print(f"強度封包: {packet_count['intensity']}")
    print(f"過濾封包: {packet_count['filtered']}")
    print(f"錯誤封包: {packet_count['error']}")
    print("="*60)
    print("數據已保存到:", DB_FILE)


if __name__ == '__main__':
    main()
