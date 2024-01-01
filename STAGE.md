 1656  dvc stage help
 1657  dvc stage add help
 1658  dvc stage add -n train -d train.py -d data \\n          -o model.h5 -o bottleneck_features_train.npy \\n          -o bottleneck_features_validation.npy -M metrics.csv \\n          python train.py
 1873  dvc stage add -n prepare \\n                -p prepare.seed,prepare.split \\n                -d src/prepare.py -d data/data.xml \\n                -o data/prepared \\n                python src/prepare.py data/data.xml
 1874  dvc stage add -n featurize \\n                -p featurize.max_features,featurize.ngrams \\n                -d src/featurization.py -d data/prepared \\n                -o data/features \\n                python src/featurization.py data/prepared data/features
 1875  dvc stage add -n train \\n                -p train.seed,train.n_est,train.min_split \\n                -d src/train.py -d data/features \\n                -o model.pkl \\n                python src/train.py data/features model.pkl
 1937  dvc stage add -n download -d scripts/data_load.py -o data/california_housing.csv python scripts/data_load.py
 1945  dvc stage add -n prepare -d scripts/prepare.py -d data/california_housing.csv -o data/california_housing_prepared.csv python scripts/prepare.py
 1948  dvc stage add -n train -d scripts/train.py -d data california_housing_prepared.csv -o models/rf_model.joblib python scripts/train.py
 1950  dvc stage add -n train -d scripts/train.py -d data/california_housing_prepared.csv -o models/rf_model.joblib python scripts/train.py
 1954  dvc stage add -n evaluate -d scripts/evaluate.py -d models/rf_model.joblib python scripts/evaluate.py
 2196* dvc stage add -n metrics -d scripts/metrics.py -d data/california_housing_prepared.csv -d models/rf_model.joblib -o metrics/mse_plot.png -o metrics/mse_values.csv python scripts/metrics.py
 2198* dvc stage remove -n metrics
 2199  dvc stage add --help
 2200* dvc stage add -n metrics -d scripts/metrics.py -d data/california_housing_prepared.csv -d models/rf_model.joblib -o metrics/mse_plot.png -m metrics/mse_values.csv python scripts/metrics.py