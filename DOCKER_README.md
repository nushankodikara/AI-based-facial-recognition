# Face Recognition API Docker Setup

This document provides instructions for containerizing and running the Face Recognition API using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- The YOLOv11n-face.pt model file in the project root directory

## Files Required for Docker

The following files are required to build the Docker image:

1. `api.py` - The main API application code
2. `requirements.txt` - Python dependencies
3. `yolov11n-face.pt` - The YOLO model file for face detection
4. `Dockerfile` - Instructions for building the Docker image
5. `docker-compose.yml` - Configuration for running the containerized application

## Directory Structure

The application uses the following directory structure, which is mapped to persistent storage:

```
.
├── api.py
├── requirements.txt
├── yolov11n-face.pt
├── Dockerfile
├── docker-compose.yml
├── faces.db
├── processed/
│   └── known/
├── unprocessed/
│   └── known/
├── uploads/
└── temp/
```

## Building and Running with Docker Compose

1. Make sure all required files are in place
2. Build and start the container:

```bash
docker-compose up -d
```

3. To stop the container:

```bash
docker-compose down
```

## Accessing the API

Once the container is running, you can access:

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Persistent Data

The following data is persisted through Docker volumes:

- Database: `./faces.db:/app/faces.db`
- Processed faces: `./processed:/app/processed`
- Unprocessed images: `./unprocessed:/app/unprocessed`
- Uploaded images: `./uploads:/app/uploads`
- Temporary files: `./temp:/app/temp`
- Logs: `./api.log:/app/api.log`

## Troubleshooting

If you encounter issues:

1. Check the logs:
```bash
docker-compose logs
```

2. Ensure the model file is present in the build context
3. Verify that all required directories exist locally
4. Check that the database file has appropriate permissions 