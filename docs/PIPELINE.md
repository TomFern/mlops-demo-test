dvc stage add -n download -d scripts/data_load.py -o data/california_housing.csv python scripts/data_load.py

# updates data/.gitignore

git add dvc.yaml dvc.lock
git commit -m "create ML pipeline"

dvc stage add -n validate -d scripts/data_validate.py -d data/california_housing.csv -m metrics/data_report.txt -o metrics/data_correlation_matrix.png -o metrics/data_histogram.png python scripts/data_validate.py

dvc stage add -n prepare -d scripts/prepare.py -d data/california_housing.csv -o data/california_housing_prepared.csv python scripts/prepare.py

dvc dag
dvc repro

dvc stage add -n train -d scripts/train.py -d data/california_housing_prepared.csv -o models/rf_model.joblib python scripts/train.py

dvc stage add -n evaluate -d scripts/evaluate.py -d models/rf_model.joblib python scripts/evaluate.py
(returns exit code 1 if error > cutoff)

dvc stage add -n metrics -d scripts/metrics.py -d data/california_housing_prepared.csv -d models/rf_model.joblib -o metrics/mse_plot.png -m metrics/mse_values.csv python scripts/metrics.py
