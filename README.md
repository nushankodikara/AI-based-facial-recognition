# AI-based Facial Recognition System

This project provides a facial recognition system with both a GUI application and a REST API.

## Features

- Face detection using YOLO (YOLOv11n-face)
- Face recognition using face_recognition library
- Database storage of known faces
- GUI application for interactive use
- REST API for programmatic access

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-based-facial-recognition.git
cd AI-based-facial-recognition
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the YOLOv11n-face model (if not already included):
```bash
# The model should be in the root directory as yolov11n-face.pt
```

## Directory Structure

- `processed/known/`: Directory for storing processed face images
- `unprocessed/known/`: Directory for storing unprocessed images
- `temp/`: Directory for temporary files
- `uploads/`: Directory for uploaded images via API

## Usage

### GUI Application

Run the GUI application:

```bash
python main.py
```

The GUI application provides:
- Processing of unprocessed images to extract faces
- Recognition of faces in new images

### REST API

Run the API server:

```bash
python api.py
```

The API server will start at http://localhost:8000 by default.

## API Documentation

Once the API is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### Person Management

- `GET /persons`: Get all persons in the database
- `GET /persons/{person_id}`: Get a specific person by ID
- `POST /persons`: Create a new person with a face image
- `PUT /persons/{person_id}`: Update a person's information
- `DELETE /persons/{person_id}`: Delete a person and their face image

#### Face Detection and Recognition

- `POST /detect-faces`: Detect faces in an image without recognition
- `POST /recognize`: Recognize faces in an image
- `POST /upload-batch`: Upload multiple images for processing

## Examples

### Adding a New Person

```bash
curl -X POST "http://localhost:8000/persons" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "name=John Doe" \
  -F "face_image=@/path/to/image.jpg"
```

### Recognizing Faces in an Image

```bash
curl -X POST "http://localhost:8000/recognize" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/image.jpg"
```

## License

[MIT License](LICENSE)
