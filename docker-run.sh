#!/bin/bash

# Script to build and run the Face Recognition API Docker container

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if required files exist
if [ ! -f "api.py" ]; then
    echo "Error: api.py file not found."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt file not found."
    exit 1
fi

if [ ! -f "yolov11n-face.pt" ]; then
    echo "Error: yolov11n-face.pt model file not found."
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found."
    exit 1
fi

if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml file not found."
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p processed/known
mkdir -p unprocessed/known
mkdir -p uploads
mkdir -p temp

# Build and start the container
echo "Building and starting the Face Recognition API container..."
docker-compose up -d --build

# Check if the container is running
if [ "$(docker ps -q -f name=face-recognition-api)" ]; then
    echo "Face Recognition API is now running!"
    echo "API is accessible at: http://localhost:8000"
    echo "API documentation is available at: http://localhost:8000/docs"
else
    echo "Error: Failed to start the container. Check the logs with 'docker-compose logs'."
    exit 1
fi 