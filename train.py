import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# MLflow 新增: 匯入 mlflow 函式庫
import mlflow
import mlflow.sklearn

# 1. 載入資料集
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    columns = ['sepal-length', 'sepal-width',
               'petal-length', 'petal-width', 'class']
    df = pd.read_csv(url, names=columns)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Data loaded successfully")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. 定義實驗參數
# MLflow 新增: 我們將 test_size 和 random_state 這兩個超參數提出來，方便記錄
test_split_ratio = 0.3
random_seed = 42

# 3. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split_ratio, random_state=random_seed)
print("Data split into training and testing sets")


# MLflow 新增: 使用 with mlflow.start_run(): 來開啟一個新的 MLflow Run
# 在這個區塊內的所有 mlflow.log_... 都會被記錄到這次 Run 當中
with mlflow.start_run():

    print("Model training completed")

    # 4. 建立並訓練模型
    # MLflow 新增: 將模型超參數也定義出來
    model_params = {"max_iter": 200}
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    # 5. 評估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    # MLflow 新增: 記錄本次實驗的參數 (Parameters)
    mlflow.log_param("test_split_ratio", test_split_ratio)
    mlflow.log_param("random_seed", random_seed)
    # 我們可以迭代一個字典來記錄所有模型參數
    for key, value in model_params.items():
        mlflow.log_param(key, value)

    # MLflow 新增: 記錄本次實驗的指標 (Metrics)
    mlflow.log_metric("accuracy", acc)

    # MLflow 新增: 記錄模型本身作為產出物 (Artifact)
    # mlflow.sklearn.log_model 會幫我們把模型和環境依賴都打包好
    # "model" 是我們給這個模型產出物取的名字
    mlflow.sklearn.log_model(model, "model")

    print("MLflow Run completed and logged.")
