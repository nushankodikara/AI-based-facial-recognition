#!/bin/bash

# Script to stop and clean up the Face Recognition API Docker container

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

# Check if the container is running
if [ "$(docker ps -q -f name=face-recognition-api)" ]; then
    echo "Stopping the Face Recognition API container..."
    docker-compose down
    echo "Container stopped successfully."
else
    echo "The Face Recognition API container is not running."
fi

# Ask if user wants to clean up volumes
read -p "Do you want to clean up temporary files? (y/n): " clean_temp
if [[ $clean_temp == "y" || $clean_temp == "Y" ]]; then
    echo "Cleaning up temporary files..."
    rm -rf temp/*
    echo "Temporary files cleaned up."
fi

read -p "Do you want to clean up uploaded files? (y/n): " clean_uploads
if [[ $clean_uploads == "y" || $clean_uploads == "Y" ]]; then
    echo "Cleaning up uploaded files..."
    rm -rf uploads/*
    echo "Uploaded files cleaned up."
fi

# Ask if user wants to remove the Docker image
read -p "Do you want to remove the Docker image? (y/n): " remove_image
if [[ $remove_image == "y" || $remove_image == "Y" ]]; then
    echo "Removing the Face Recognition API Docker image..."
    docker rmi $(docker images -q face-recognition-api_face-recognition-api)
    echo "Docker image removed."
fi

echo "Cleanup completed." 