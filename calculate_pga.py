"""
計算 earthquake_data.db 中 sensor_data 表的 PGA
PGA = sqrt(X^2 + Y^2 + Z^2)
"""

import sqlite3
import math
from datetime import datetime, timezone

DB_FILE = 'earthquake_data.db'
OUTPUT_FILE = 'pga_output.txt'
CSV_OUTPUT_FILE = 'pga_output.csv'


def load_sensor_data():
    """從資料庫載入感測器資料"""
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute('''
            SELECT id, timestamp_ms, x, y, z, received_time
            FROM sensor_data
            ORDER BY timestamp_ms ASC
        ''').fetchall()


def format_timestamp(timestamp_ms):
    """13 位毫秒時間戳轉為可讀字串，否則原樣輸出"""
    if timestamp_ms >= 1000000000000:
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return f"{timestamp_ms}"


def process_and_calculate_pga(sensor_rows):
    """處理資料並計算 PGA"""
    results = []
    for row_id, timestamp_ms, x, y, z, received_time in sensor_rows:
        results.append({
            'id': row_id,
            'timestamp_ms': timestamp_ms,
            'timestamp_str': format_timestamp(timestamp_ms),
            'x': x,
            'y': y,
            'z': z,
            'pga': math.hypot(x, y, z),  # sqrt(x^2 + y^2 + z^2)
            'received_time': received_time
        })
    return results


def save_to_text(results):
    """保存為文字格式"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PGA 計算結果\n")
        f.write("="*80 + "\n")
        f.write(f"總筆數: {len(results)}\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'ID':<8} {'時間戳記':<25} {'X(Gal)':<12} {'Y(Gal)':<12} {'Z(Gal)':<12} {'PGA(Gal)':<12}\n")
        f.write("-" * 80 + "\n")

        for r in results:
            f.write(f"{r['id']:<8} {r['timestamp_str']:<25} "
                   f"{r['x']:>11.4f} {r['y']:>11.4f} {r['z']:>11.4f} {r['pga']:>11.4f}\n")

    print(f"✓ 文字格式結果已保存到: {OUTPUT_FILE}")


def save_to_csv(results):
    """保存為 CSV 格式"""
    with open(CSV_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("ID,Timestamp(ms),Timestamp(UTC),X(Gal),Y(Gal),Z(Gal),PGA(Gal),Received_Time\n")
        for r in results:
            f.write(f"{r['id']},{r['timestamp_ms']},{r['timestamp_str']},"
                   f"{r['x']:.6f},{r['y']:.6f},{r['z']:.6f},{r['pga']:.6f},"
                   f"{r['received_time']}\n")

    print(f"✓ CSV 格式結果已保存到: {CSV_OUTPUT_FILE}")


def print_statistics(results):
    """打印統計資訊"""
    if not results:
        print("沒有數據可統計")
        return

    print("\n" + "="*60)
    print("統計資訊")
    print("="*60)
    print(f"總筆數: {len(results)}")

    for label, key in [('X 軸', 'x'), ('Y 軸', 'y'), ('Z 軸', 'z'), ('PGA', 'pga')]:
        values = [r[key] for r in results]
        n = len(values)
        m = sum(values) / n
        std = math.sqrt(sum((v - m) ** 2 for v in values) / n)
        print(f"\n{label} (Gal):")
        print(f"  最小值: {min(values):.4f}")
        print(f"  最大值: {max(values):.4f}")
        print(f"  平均值: {m:.4f}")
        print(f"  標準差: {std:.4f}")
    print("="*60)


def main():
    """主程式"""
    print("QuakeWatch - 計算 PGA")
    print("="*60)
    print(f"資料庫: {DB_FILE}")
    print("="*60)

    print("\n正在從資料庫載入資料...")
    sensor_rows = load_sensor_data()
    print(f"✓ 已載入 {len(sensor_rows)} 筆資料")

    if not sensor_rows:
        print("⚠ 資料庫中沒有資料")
        return

    print("\n正在計算 PGA...")
    results = process_and_calculate_pga(sensor_rows)
    print(f"✓ 已計算 {len(results)} 筆 PGA")

    save_to_text(results)
    save_to_csv(results)
    print_statistics(results)

    print("\n前10筆數據預覽:")
    print("-" * 80)
    print(f"{'ID':<8} {'時間戳記':<25} {'X(Gal)':<12} {'Y(Gal)':<12} {'Z(Gal)':<12} {'PGA(Gal)':<12}")
    print("-" * 80)
    for r in results[:10]:
        print(f"{r['id']:<8} {r['timestamp_str']:<25} "
              f"{r['x']:>11.4f} {r['y']:>11.4f} {r['z']:>11.4f} {r['pga']:>11.4f}")

    print("\n" + "="*60)
    print("處理完成！")
    print("="*60)


if __name__ == '__main__':
    main()
