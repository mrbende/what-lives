#!/bin/bash

### LAUNCH JUPYTER LAB
jupyter lab \
    --ServerApp.allow_origin='*' \
    --ip="0.0.0.0" \
    --ServerApp.token="" \
    --no-browser \
    --notebook-dir=/workspace/what-lives \
    --LabApp.default_theme='dark'
