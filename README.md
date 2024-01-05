# MLOps CI/CD Demo

This repository demoes CI/CD automation for ML. The project uses [dvc](https://dvc.org) to build and test a model in a reproducible way.

## Setup

1. Fork and clone this repository.
2. Install [dvc](https://dvc.org) on your machine. Check you have Python installed.
3. Create a virtualenv:
    ```bash
    $ python -m venv venv
    $ source venv/bin/activate
    ```
4. Install dependencies:
    ```bash
    $ pip install -r requirements.txt

## Training the model

To download the training dataset and train the model run:

```bash
$ dvc repro
```

This will create the model file in the `models` directory

You can try the model's inference with:

```bash
python scripts/predict.py
```

The `metrics` should now contain benchmarks of the model in plot/png and CSV formats.

## ML Stages

The DVC pipeline contains the following stages:
1. download: download traininig dataset
2. validate: validate dataset
3. prepare: prepare dataset
4. train: train the model
5. evaluate: evaluate model
6. metrics: calculate model error

## API Server

The project includes an Flask API server. You can start it with

```bash
$ cd api
$ python server.py
```

## Docker image

You can build a Docker image with the model and API server with:

```bash
$ docker build -t california-housing .
```

Try running the build image with:

```bash
$ docker run -it -p 8080:8080 california-housing
```

## Deploy to Fly.io

To create the app for the first time:
1. Create a free Fly.io account
2. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/)
3. Run `fly launch`

To deploy the app with CI/CD


