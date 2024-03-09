# %%
import yaml

from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from omegaconf import DictConfig

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
## Experimentの内容を参照する
################
client = MlflowClient(tracking_uri=config.tracking_server_uri)
# 全てのExperimentの参照
all_experiments = client.search_experiments()
print("全てのExperimentの参照")
for ex in client.search_experiments():
    print("--", ex.name)
    pprint(ex)

# ある実験の参照
target_exp_name = "Default"
print(f"Experiment{target_exp_name}の参照")
default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == target_exp_name
][0]
pprint(default_experiment)

# %%
################
## experimentに情報をつけて生成する
################
# experimentの説明を生成
experiment_description = (
    "This is the grocery forecasting project. "
    "This experiment contains the produce models for apples."
)
# 検索のためのタグを追加
experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
}
# experimentを作成
produce_apples_experiment = client.create_experiment(
    name="Apple_Models", tags=experiment_tags
)

# %%
################
## タグに基づいてexperimentを検索する
################
search_key = "tags.`project_name`"
search_val = "'grocery-forecasting'"
apples_experiment = client.search_experiments(
    filter_string=f"{search_key} = {search_val}"
)
print(vars(apples_experiment[0]))
# %%
apples_experiment[0]
# %%
vars(apples_experiment[0])
# %%
