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


def calculate_pga(x, y, z):
    """計算 PGA = sqrt(X^2 + Y^2 + Z^2)"""
    return math.sqrt(x**2 + y**2 + z**2)


def load_sensor_data():
    """從資料庫載入感測器資料"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, timestamp_ms, x, y, z, received_time
        FROM sensor_data
        ORDER BY timestamp_ms ASC
    ''')
    sensor_rows = cursor.fetchall()
    conn.close()

    return sensor_rows


def process_and_calculate_pga(sensor_rows):
    """處理資料並計算 PGA"""
    results = []
    
    for row in sensor_rows:
        row_id, timestamp_ms, x, y, z, received_time = row
        
        # 計算 PGA
        pga = calculate_pga(x, y, z)
        
        # 轉換時間戳記為可讀格式
        if timestamp_ms >= 1000000000000:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        else:
            timestamp_str = f"{timestamp_ms}"
        
        results.append({
            'id': row_id,
            'timestamp_ms': timestamp_ms,
            'timestamp_str': timestamp_str,
            'x': x,
            'y': y,
            'z': z,
            'pga': pga,
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
        
        # 寫入標題
        f.write(f"{'ID':<8} {'時間戳記':<25} {'X(Gal)':<12} {'Y(Gal)':<12} {'Z(Gal)':<12} {'PGA(Gal)':<12}\n")
        f.write("-" * 80 + "\n")
        
        # 寫入數據
        for r in results:
            f.write(f"{r['id']:<8} {r['timestamp_str']:<25} "
                   f"{r['x']:>11.4f} {r['y']:>11.4f} {r['z']:>11.4f} {r['pga']:>11.4f}\n")
    
    print(f"✓ 文字格式結果已保存到: {OUTPUT_FILE}")


def save_to_csv(results):
    """保存為 CSV 格式"""
    with open(CSV_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 寫入標題
        f.write("ID,Timestamp(ms),Timestamp(UTC),X(Gal),Y(Gal),Z(Gal),PGA(Gal),Received_Time\n")
        
        # 寫入數據
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
    
    x_values = [r['x'] for r in results]
    y_values = [r['y'] for r in results]
    z_values = [r['z'] for r in results]
    pga_values = [r['pga'] for r in results]
    
    def mean(values):
        return sum(values) / len(values) if values else 0
    
    def std(values):
        if not values:
            return 0
        m = mean(values)
        variance = sum((x - m) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    print("\n" + "="*60)
    print("統計資訊")
    print("="*60)
    print(f"總筆數: {len(results)}")
    
    print(f"\nX 軸 (Gal):")
    print(f"  最小值: {min(x_values):.4f}")
    print(f"  最大值: {max(x_values):.4f}")
    print(f"  平均值: {mean(x_values):.4f}")
    print(f"  標準差: {std(x_values):.4f}")
    
    print(f"\nY 軸 (Gal):")
    print(f"  最小值: {min(y_values):.4f}")
    print(f"  最大值: {max(y_values):.4f}")
    print(f"  平均值: {mean(y_values):.4f}")
    print(f"  標準差: {std(y_values):.4f}")
    
    print(f"\nZ 軸 (Gal):")
    print(f"  最小值: {min(z_values):.4f}")
    print(f"  最大值: {max(z_values):.4f}")
    print(f"  平均值: {mean(z_values):.4f}")
    print(f"  標準差: {std(z_values):.4f}")
    
    print(f"\nPGA (Gal):")
    print(f"  最小值: {min(pga_values):.4f}")
    print(f"  最大值: {max(pga_values):.4f}")
    print(f"  平均值: {mean(pga_values):.4f}")
    print(f"  標準差: {std(pga_values):.4f}")
    print("="*60)


def main():
    """主程式"""
    print("QuakeWatch - 計算 PGA")
    print("="*60)
    print(f"資料庫: {DB_FILE}")
    print("="*60)
    
    # 載入資料
    print("\n正在從資料庫載入資料...")
    sensor_rows = load_sensor_data()
    print(f"✓ 已載入 {len(sensor_rows)} 筆資料")
    
    if len(sensor_rows) == 0:
        print("⚠ 資料庫中沒有資料")
        return
    
    # 計算 PGA
    print("\n正在計算 PGA...")
    results = process_and_calculate_pga(sensor_rows)
    print(f"✓ 已計算 {len(results)} 筆 PGA")
    
    # 保存結果
    save_to_text(results)
    save_to_csv(results)
    
    # 打印統計資訊
    print_statistics(results)
    
    # 顯示前10筆數據預覽
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
