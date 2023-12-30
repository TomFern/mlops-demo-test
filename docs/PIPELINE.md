dvc stage add -n download -d scripts/data_load.py -o data/california_housing.csv python scripts/data_load.py

# updates data/.gitignore

git add dvc.yaml dvc.lock
git commit -m "create ML pipeline"

dvc stage add -n prepare -d scripts/prepare.py -d data/california_housing.csv -o data/california_housing_prepared.csv python scripts/prepare.py

dvc dag
dvc repro

dvc stage add -n train -d scripts/train.py -d data/california_housing_prepared.csv -o models/rf_model.joblib python scripts/train.py

dvc stage add -n evaluate -d scripts/evaluate.py -d models/rf_model.joblib python scripts/evaluate.py
(returns exit code 1 if error > cutoff)
