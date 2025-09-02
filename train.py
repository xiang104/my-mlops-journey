import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. 載入資料集
# 我們使用一個經典的公開資料集：鳶尾花 (Iris)
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = pd.read_csv(url, names=columns)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Data loaded successfully")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets")

# 3. 建立並訓練模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model training completed")

# 4. 評估模型
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# 5. 儲存訓練好的模型
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")