# 1. 選擇一個基礎映像檔
# 我們選擇官方的 python 3.9 slim 版本，它是一個輕量化的 Linux 環境
FROM python:3.10-slim

# 2. 設定工作目錄
# 在容器內建立一個 /app 的資料夾，並將後續指令都在此資料夾內執行
WORKDIR /app

# 3. 複製依賴清單
# 將我們本地的 requirements.txt 複製到容器內的 /app 資料夾下
COPY requirements.txt .

# 4. 安裝依賴
# 在容器內執行 pip install 指令來安裝所有函式庫
# --no-cache-dir 是一個好習慣，可以讓映像檔小一點
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製你的應用程式程式碼
# 將所有本地檔案 (用 . 代表) 複製到容器內的 /app 資料夾下
COPY . .

# 6. 設定容器啟動時要執行的指令
# 當容器啟動時，預設會執行 `python3 train.py`
CMD ["python", "train.py"]