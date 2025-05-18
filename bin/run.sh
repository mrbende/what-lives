#!/bin/bash

#### CONTAINER RUN PARAMS
IMAGE_NAME="what-lives"
TAG="main"
PORT=8888

echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build -t "$IMAGE_NAME:$TAG" .

if [ $? -eq 0 ]; then
    echo "Successfully built Docker image"
    echo "Starting container on port $PORT"
    docker run -it \
      -p "$PORT:$PORT" \
      -v "$(pwd):/workspace/what-lives" \
      "$IMAGE_NAME:$TAG" /bin/bash
    
    if [ $? -eq 0 ]; then
        echo "Container started successfully"
    else
        echo "Error: Failed to start container"
        exit 1
    fi
else
    echo "Error: Failed to build Docker image"
    exit 1
fi
