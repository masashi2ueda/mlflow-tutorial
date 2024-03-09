# %%
import mlflow
import yaml
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf


# %%
################
## 設定を読み込む
################
with open('../config/config.yml', 'r') as yml:
    config = yaml.safe_load(yml)
config = DictConfig(config)
config.tracking_server_uri
config

# %%
################
## 仮のモデルを学習する
################
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# %%
################
## mlflowに登録する
################
# トラッキングサーバを指定
mlflow.set_tracking_uri(uri=config.tracking_server_uri)
# 実験管理名を指定
mlflow.set_experiment(config.experiment_name)
# 実験を開始
with mlflow.start_run():
    # パラメータを保存
    mlflow.log_params(params)
    # 指標を保存
    mlflow.log_metric("accuracy", accuracy)
    # タグ情報を追加
    mlflow.set_tag("Training Info1", "学習タグ情報1")
    mlflow.set_tag("Training Info2", "学習タグ情報2")
    # データの型と形を保存する
    signature = infer_signature(X_train, lr.predict(X_train))
    # モデルを保存する
    model_info = mlflow.sklearn.log_model(
        sk_model=lr, # sklearnのモデルはそのまま登録
        artifact_path="iris_model", # 保存する場所の名前
        signature=signature, # データの型
        input_example=X_train, # 学習データの例
        registered_model_name="tracking-quickstart", # モデル名
    )
    # model uriを保存しておく（webサーバのどこからとってこればよいかわからない）
    mlflow.log_params({"model_uri":model_info.model_uri})

# %%
################
## 再利用する
################
# データを作成
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# モデルを読み込み
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
print("model_info.model_uri:", model_info.model_uri)

# 結果を確認
predictions = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
result[:4]