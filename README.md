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

