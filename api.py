from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import os
import cv2
import numpy as np
import face_recognition
import shutil
import uuid
from ultralytics import YOLO
import base64
from datetime import datetime
import threading
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger("face_recognition_api")

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define directories
DIRS = {
    'uploads': 'uploads',
    'processed': 'processed/known',
    'temp': 'temp'
}

# Create directories if they don't exist
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Mount static files directory for serving face images
app.mount("/faces", StaticFiles(directory=DIRS['processed']), name="faces")

# SQLite connection pool
class SQLiteConnectionPool:
    def __init__(self, database, max_connections=10):
        self.database = database
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
    
    @contextmanager
    def get_connection(self):
        with self.lock:
            if not self.connections:
                # Create a new connection if none are available
                connection = sqlite3.connect(self.database, check_same_thread=False)
                connection.row_factory = sqlite3.Row
            else:
                # Reuse an existing connection
                connection = self.connections.pop()
        
        try:
            yield connection
        finally:
            with self.lock:
                if len(self.connections) < self.max_connections:
                    # Return the connection to the pool
                    self.connections.append(connection)
                else:
                    # Close the connection if the pool is full
                    connection.close()
    
    def close_all(self):
        with self.lock:
            for connection in self.connections:
                connection.close()
            self.connections = []

# Create connection pool
db_pool = SQLiteConnectionPool('faces.db')

# Database connection dependency
def get_db():
    with db_pool.get_connection() as conn:
        yield conn

# Initialize YOLO model
model = YOLO('yolov11n-face.pt')

# Pydantic models for request/response
class Person(BaseModel):
    id: int
    name: str
    image_path: str

class PersonCreate(BaseModel):
    name: str

class PersonUpdate(BaseModel):
    name: str

class RecognitionResult(BaseModel):
    faces: List[dict]
    image_path: Optional[str] = None

# Helper functions
def setup_database(conn):
    """Initialize SQLite database"""
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_path TEXT NOT NULL
        )
    ''')
    conn.commit()

def get_next_person_id(conn):
    """Get the next available person ID"""
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(id) FROM persons')
    result = cursor.fetchone()[0]
    return 1 if result is None else result + 1

def process_image(image_data, conn):
    """Process image data to detect and extract faces"""
    logger.info("Processing image to detect faces")
    # Convert image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logger.error("Could not read image data")
        raise HTTPException(status_code=400, detail="Could not read image")
    
    # Run face detection
    logger.info("Running face detection with YOLO model")
    results = model(img)
    
    faces = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        if len(boxes) == 0:
            logger.info("No faces detected in the image")
            continue
        
        # Process each detected face
        logger.info(f"Found {len(boxes)} faces in the image")
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop face
            face_img = img[y1:y2, x1:x2]
            
            # Generate a unique filename
            face_filename = f"face_{uuid.uuid4()}.jpg"
            temp_face_path = os.path.join(DIRS['temp'], face_filename)
            
            # Save face temporarily
            cv2.imwrite(temp_face_path, face_img)
            logger.debug(f"Saved temporary face image to {temp_face_path}")
            
            faces.append({
                "id": i,
                "bbox": [x1, y1, x2, y2],
                "temp_path": temp_face_path
            })
    
    return faces, img

def recognize_face(face_encoding, conn):
    """Recognize a face by comparing with known faces"""
    logger.info("Recognizing face against known faces")
    cursor = conn.cursor()
    
    # Get face encodings from database
    known_face_encodings = []
    known_face_data = []
    
    # Query database for processed faces
    cursor.execute('SELECT id, name, image_path FROM persons')
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} known faces in database")
    
    for row in rows:
        person_id, name, face_path = row['id'], row['name'], row['image_path']
        try:
            known_img = face_recognition.load_image_file(face_path)
            known_encoding = face_recognition.face_encodings(known_img)[0]
            known_face_encodings.append(known_encoding)
            known_face_data.append({
                "id": person_id,
                "name": name,
                "image_path": face_path
            })
        except Exception as e:
            logger.error(f"Error loading face {name}: {str(e)}")
    
    if not known_face_encodings:
        logger.warning("No known face encodings available for comparison")
        return None, 0.0
    
    # Compare with known faces
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = int(np.argmin(face_distances))
    
    if face_distances[best_match_index] < 0.6:  # Threshold for a match
        confidence = 1 - face_distances[best_match_index]
        logger.info(f"Face recognized as {known_face_data[best_match_index]['name']} with confidence {confidence:.2f}")
        return known_face_data[best_match_index], confidence
    
    logger.info("Face not recognized (below confidence threshold)")
    return None, 0.0

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    with db_pool.get_connection() as conn:
        setup_database(conn)

# Close all database connections on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    db_pool.close_all()

# API Endpoints
@app.get("/")
async def root():
    logger.info("API health check")
    return {"message": "Face Recognition API is running"}

@app.get("/persons", response_model=List[Person])
async def get_all_persons(conn = Depends(get_db)):
    """Get all persons in the database"""
    logger.info("Received request to get all persons")
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, image_path FROM persons')
        persons = [dict(row) for row in cursor.fetchall()]
        
        # Convert image paths to URLs
        for person in persons:
            person['image_path'] = f"/faces/{os.path.basename(person['image_path'])}"
        
        logger.info(f"Returning {len(persons)} persons")
        return persons
    except Exception as e:
        logger.error(f"Error getting all persons: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/persons/{person_id}", response_model=Person)
async def get_person(person_id: int, conn = Depends(get_db)):
    """Get a specific person by ID"""
    logger.info(f"Received request to get person with ID: {person_id}")
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, image_path FROM persons WHERE id = ?', (person_id,))
        person = cursor.fetchone()
        
        if not person:
            logger.warning(f"Person with ID {person_id} not found")
            raise HTTPException(status_code=404, detail="Person not found")
        
        person_dict = dict(person)
        person_dict['image_path'] = f"/faces/{os.path.basename(person_dict['image_path'])}"
        
        logger.info(f"Returning person: {person_dict['name']}")
        return person_dict
    except Exception as e:
        logger.error(f"Error getting person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/persons", response_model=Person)
async def create_person(
    name: str = Form(...),
    face_image: UploadFile = File(...),
    conn = Depends(get_db)
):
    """Create a new person with a face image"""
    logger.info(f"Received request to create person '{name}' with image: {face_image.filename}")
    faces = []
    try:
        # Read uploaded image
        image_data = await face_image.read()
        
        # Process image to detect faces
        faces, _ = process_image(image_data, conn)
        
        if not faces:
            logger.warning("No faces detected in the uploaded image")
            raise HTTPException(status_code=400, detail="No faces detected in the image")
        
        # Use the first detected face
        face = faces[0]
        
        # Get next person ID
        person_id = get_next_person_id(conn)
        
        # Create permanent filename
        face_filename = f"person_{person_id}.jpg"
        face_path = os.path.join(DIRS['processed'], face_filename)
        
        # Move the temporary face file to permanent location
        shutil.move(face["temp_path"], face_path)
        logger.info(f"Saved face image to {face_path}")
        
        # Save to database
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO persons (id, name, image_path) VALUES (?, ?, ?)',
            (person_id, name, face_path)
        )
        conn.commit()
        logger.info(f"Added person to database with ID: {person_id}")
        
        # Mark this face as processed so it doesn't get cleaned up in the finally block
        face["processed"] = True
        
        return {
            "id": person_id,
            "name": name,
            "image_path": f"/faces/{face_filename}"
        }
    
    except Exception as e:
        logger.error(f"Error creating person: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up any remaining temporary files
        for f in faces:
            try:
                # Skip the face that was successfully processed
                if f.get("processed", False):
                    continue
                    
                if os.path.exists(f["temp_path"]):
                    os.remove(f["temp_path"])
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")

@app.put("/persons/{person_id}", response_model=Person)
async def update_person(
    person_id: int,
    person_update: PersonUpdate,
    conn = Depends(get_db)
):
    """Update a person's information"""
    logger.info(f"Received request to update person {person_id} to name: {person_update.name}")
    try:
        cursor = conn.cursor()
        
        # Check if person exists
        cursor.execute('SELECT id FROM persons WHERE id = ?', (person_id,))
        if not cursor.fetchone():
            logger.warning(f"Person with ID {person_id} not found")
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Update person
        cursor.execute(
            'UPDATE persons SET name = ? WHERE id = ?',
            (person_update.name, person_id)
        )
        conn.commit()
        
        # Get updated person
        cursor.execute('SELECT id, name, image_path FROM persons WHERE id = ?', (person_id,))
        person = dict(cursor.fetchone())
        person['image_path'] = f"/faces/{os.path.basename(person['image_path'])}"
        
        logger.info(f"Successfully updated person {person_id}")
        return person
    except Exception as e:
        logger.error(f"Error updating person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/persons/{person_id}")
async def delete_person(person_id: int, conn = Depends(get_db)):
    """Delete a person and their face image"""
    logger.info(f"Received request to delete person with ID: {person_id}")
    try:
        cursor = conn.cursor()
        
        # Get person's image path
        cursor.execute('SELECT image_path FROM persons WHERE id = ?', (person_id,))
        person = cursor.fetchone()
        
        if not person:
            logger.warning(f"Person with ID {person_id} not found")
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Delete image file
        image_path = person['image_path']
        if os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Deleted face image: {image_path}")
        
        # Delete from database
        cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
        conn.commit()
        
        logger.info(f"Successfully deleted person {person_id}")
        return {"message": "Person deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting person {person_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-faces")
async def detect_faces(
    image: UploadFile = File(...),
    conn = Depends(get_db)
):
    """Detect faces in an image without recognition"""
    logger.info(f"Received face detection request for image: {image.filename}")
    faces = []
    try:
        # Read uploaded image
        image_data = await image.read()
        
        # Process image to detect faces
        faces, img = process_image(image_data, conn)
        
        if not faces:
            logger.info("No faces detected in the uploaded image")
            return {"faces": []}
        
        # Save the image with bounding boxes
        result_filename = f"detect_{uuid.uuid4()}.jpg"
        result_path = os.path.join(DIRS['temp'], result_filename)
        
        # Draw bounding boxes
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite(result_path, img)
        
        logger.info(f"Detection completed. Found {len(faces)} faces.")
        # Return face information
        return {
            "faces": [{"bbox": face["bbox"]} for face in faces],
            "result_image": f"/temp/{result_filename}"
        }
    
    except Exception as e:
        logger.error(f"Error during face detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary face files
        for face in faces:
            try:
                if os.path.exists(face["temp_path"]):
                    os.remove(face["temp_path"])
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")

@app.post("/recognize", response_model=RecognitionResult)
async def recognize_faces_endpoint(
    image: UploadFile = File(...),
    conn = Depends(get_db)
):
    """Recognize faces in an image"""
    logger.info(f"Received face recognition request for image: {image.filename}")
    faces = []
    try:
        # Read uploaded image
        image_data = await image.read()
        
        # Process image to detect faces
        faces, img = process_image(image_data, conn)
        
        if not faces:
            logger.info("No faces detected in the uploaded image")
            return {"faces": []}
        
        # Recognize each face
        recognition_results = []
        for face in faces:
            # Get face encoding
            face_img = cv2.imread(face["temp_path"])
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if not face_encodings:
                # Skip if no encoding could be generated
                logger.warning(f"Could not generate encoding for face at {face['temp_path']}")
                recognition_results.append({
                    "bbox": face["bbox"],
                    "name": "Unknown",
                    "confidence": 0.0,
                    "person_id": None
                })
                continue
            
            face_encoding = face_encodings[0]
            
            # Recognize face
            person_data, confidence = recognize_face(face_encoding, conn)
            
            if person_data:
                # Draw rectangle with name
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{person_data['name']} ({confidence:.2f})", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                recognition_results.append({
                    "bbox": face["bbox"],
                    "name": person_data["name"],
                    "confidence": float(confidence),
                    "person_id": person_data["id"]
                })
            else:
                # Unknown face
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                recognition_results.append({
                    "bbox": face["bbox"],
                    "name": "Unknown",
                    "confidence": 0.0,
                    "person_id": None
                })
        
        # Save the result image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"recognize_{timestamp}.jpg"
        result_path = os.path.join(DIRS['temp'], result_filename)
        cv2.imwrite(result_path, img)
        
        logger.info(f"Recognition completed. Found {len(recognition_results)} faces.")
        return {
            "faces": recognition_results,
            "image_path": f"/temp/{result_filename}"
        }
    
    except Exception as e:
        logger.error(f"Error during face recognition: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary face files
        for face in faces:
            try:
                if os.path.exists(face["temp_path"]):
                    os.remove(face["temp_path"])
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")

@app.post("/upload-batch")
async def upload_batch(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    conn = Depends(get_db)
):
    """Upload multiple images for processing"""
    logger.info(f"Received batch upload request with {len(images)} images")
    # Create a temporary directory for batch uploads
    batch_dir = os.path.join(DIRS['uploads'], f"batch_{uuid.uuid4()}")
    os.makedirs(batch_dir, exist_ok=True)
    
    saved_files = []
    
    try:
        # Save all uploaded files
        for image in images:
            file_path = os.path.join(batch_dir, image.filename)
            with open(file_path, "wb") as f:
                f.write(await image.read())
            saved_files.append(file_path)
            logger.info(f"Saved uploaded file: {file_path}")
        
        logger.info(f"Successfully uploaded {len(saved_files)} images to {batch_dir}")
        # Return the batch ID for later processing
        return {
            "message": f"Uploaded {len(saved_files)} images",
            "batch_dir": batch_dir,
            "files": saved_files
        }
    
    except Exception as e:
        logger.error(f"Error during batch upload: {str(e)}", exc_info=True)
        # Clean up in case of error
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files directory for temporary files
app.mount("/temp", StaticFiles(directory=DIRS['temp']), name="temp")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 