# Queue experiments, run them in parallel

## Extra files

The following files are changed to work with experiments:
- params.yaml: was added with the default value for parameters
- scripts/train.py: was modified to read default values from params.yaml

## Setup

1. Create branch (e.g experiments)
1. Create params.yaml
2. Set params for each stage we want to experiment
3. Add pyyaml with pip, read params.yaml and use the values in the scripts
4. Update stage so it uses params

```bash
dvc stage add -n train --force \
  -p n_estimators,random_state,test_size \
  -d scripts/train.py -d data/california_housing_prepared.csv \
  -o models/rf_model.joblib \
  python src/train.py
```

5. Run `dvc repro` to ensure everythin is ok
6. Run an experiment: `dvc exp run -S "train.n_estimators=0.1"`
7.

## Running experiments

To train with a different set of parameters:

```bash
dvc exp run -S "train.n_estimators=120" -S "train.random_state=10" -S "train.test_size=0.3
```

## CI changes

We need a conditional block. Idea to test: on main run the dvc repro normally, on branches, run an alternative block that executes all the pending experiments, do metrics change if we have more than 1?

## DVC Workflow changes

original (without parameters):

Original dag:

```
          +----------+
          | download |
          +----------+
           **        **
         **            **
        *                *
+----------+         +---------+
| validate |         | prepare |
+----------+         +---------+
                     **        **
                   **            **
                  *                **
            +-------+                *
            | train |*               *
            +-------+ ***            *
                *        ****        *
                *            ***     *
                *               **   *
          +----------+         +---------+
          | evaluate |         | metrics |
          +----------+         +---------+
```

With parameters:

