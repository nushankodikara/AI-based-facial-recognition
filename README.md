# Face Recognition Application Documentation

## Overview

This application is a facial recognition system built using Python that combines YOLOv8 for face detection and the `face_recognition` library for face recognition. It features a graphical user interface built with Tkinter for easy interaction.

## Technologies Used

- Python 3.12
- YOLOv8
- face_recognition
- SQLite
- OpenCV
- Tkinter
- PIL (Python Imaging Library)


## Prerequisites

```bash
pip install -r requirements
```


## Project Structure
```
project/
├── main.py           # Main application file
├── faces.db          # SQLite database
├── unprocessed/      # Directory for unprocessed images
│   └── known/        # Known faces to be processed
└── processed/        # Directory for processed face images
    └── known/        # Processed and cropped faces
```

## Features

### 1. Face Processing
- Processes images from the unprocessed/known directory
- Supports multiple image formats (jpg, png, webp, bmp, tiff, gif)
- Detects multiple faces in a single image
- Asks for names individually for each detected face
- Stores processed faces in the database with unique IDs
- Prevents reprocessing of already processed images


### 2. Face Recognition
- File picker to select images for recognition
- Detects faces using YOLOv8
- Matches faces against the database of known faces
- Displays results with color-coded boxes
- Shows a legend with names and confidence scores
- Scrollable legend for multiple detections


## Usage


1. **Initial Setup**
```python
app = FaceProcessor()
app.run()
```

2. **Processing Known Faces**

- Place images containing known faces in `unprocessed/known/`
- Click "Process Unprocessed Images"
- Enter names when prompted for each detected face

3. **Recognizing Faces**
- Click "Recognize Faces"
- Select an image using the file picker
- View results with color-coded boxes and legend


## Future Improvements
- Add batch processing capabilities
- Implement face recognition confidence threshold settings
- Add export/import functionality for the face database
- Add support for real-time video recognition