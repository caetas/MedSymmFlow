#!/usr/bin/env bash

docker run --rm --gpus all --ipc=host \
    --env-file ../.env \
    -v $(pwd)/../:/app/ \
    medsymmflow /bin/bash scripts/main.sh