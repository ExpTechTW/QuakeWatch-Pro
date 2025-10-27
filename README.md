# QuakeWatch - ES-Net 地震監測系統

## 功能概述

此系統分為兩個獨立程序：

1. **data_collector.py** - 數據收集器

   - 從 ESP32 接收串列埠數據
   - 執行 XOR 校驗
   - 將數據存入 SQLite 數據庫

2. **visualization.py** - 數據視覺化
   - 從 SQLite 數據庫讀取數據
   - 顯示 5 個實時圖表
   - 應用 JMA 濾波器
   - 進行 FFT 頻譜分析

## 安裝

```bash
# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

## 使用方法

### 1. 收集數據

在終端 1 運行數據收集器：

```bash
python3 data_collector.py
```

選擇串列埠，程序將開始收集數據並存入 `earthquake_data.db`。

### 2. 顯示圖表

在終端 2 運行視覺化程序：

```bash
python3 visualization.py
```

將顯示 5 個實時圖表：

- 圖表 1: 三軸加速度
- 圖表 2: 三軸濾波後加速度
- 圖表 3: 三軸頻譜（未濾波）
- 圖表 4: 三軸頻譜（濾波後）
- 圖表 5: PGA + 計測震度

## 數據格式

### 感測器數據（50Hz）

- 格式: `[0x53][timestamp: 8 bytes][X: 4 bytes][Y: 4 bytes][Z: 4 bytes][XOR: 1 byte]`
- 數據表: `sensor_data`
- 欄位: `timestamp_ms, x, y, z, received_time`

### 強度數據（2Hz）

- 格式: `[0x49][timestamp: 8 bytes][intensity: 4 bytes][a: 4 bytes][XOR: 1 byte]`
- 數據表: `intensity_data`
- 欄位: `timestamp_ms, intensity, a, received_time`

## 數據庫

所有數據保存在 `earthquake_data.db` SQLite 數據庫中。

### 自動清理

程序會自動清理超過 24 小時的舊數據。

### 查詢數據

您可以使用任何 SQLite 工具查詢數據：

```bash
sqlite3 earthquake_data.db
```

```sql
-- 查詢最近的感測器數據
SELECT * FROM sensor_data ORDER BY received_time DESC LIMIT 100;

-- 查詢最近的強度數據
SELECT * FROM intensity_data ORDER BY received_time DESC LIMIT 10;

-- 統計數據
SELECT COUNT(*) FROM sensor_data;
SELECT COUNT(*) FROM intensity_data;
```

## 優化特性

- **降低 FFT 計算頻率**: 每 300ms 更新一次（而非每 50ms）
- **批量數據更新**: 使用 deque.extend() 提高效率
- **float32 優化**: 節省 50% 記憶體使用
- **位運算優化**: 快速計算 2 的冪次方
- **數據庫索引**: 加快查詢速度

## 故障排除

### 找不到串列埠

- 檢查 ESP32 是否正確連接
- 檢查系統識別了哪個串列埠

### 數據庫文件不存在

- 先運行 `data_collector.py` 創建數據庫

### 圖表不更新

- 確認 `data_collector.py` 正在運行
- 檢查是否有數據寫入數據庫

## 系統要求

- Python 3.8+
- macOS/Linux/Windows
- 可用的串列埠
- 3GB RAM（推薦）

## 授權

MIT License
