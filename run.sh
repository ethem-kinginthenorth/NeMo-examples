#!/usr/bin/env bash

gcsfuse --only-dir Data \
        --implicit-dirs square-nemo-workshop /workspace/data

jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''