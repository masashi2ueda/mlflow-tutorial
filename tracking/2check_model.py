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
config.model_uri = "runs:/1229a42554f245d5b360eb0dec6b0d91/iris_model"

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
loaded_model = mlflow.pyfunc.load_model(config.model_uri)

# 結果を確認
predictions = loaded_model.predict(X_test)
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
result[:4]