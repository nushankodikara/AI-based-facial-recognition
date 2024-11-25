from ultralytics import YOLO
import cv2
import os
import tkinter as tk
from tkinter import messagebox
import sqlite3
from PIL import Image
import numpy as np
from PIL import Image, ImageTk
from PIL import ImageDraw
import random

class FaceProcessor:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov11n-face.pt')
        
        # Setup directories
        self.dirs = {
            'unprocessed': 'unprocessed/known',
            'processed': 'processed/known'
        }
        self.create_directories()
        
        # Setup database
        self.setup_database()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Face Processing Application")
        self.root.geometry("800x600")
        self.setup_gui()

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def setup_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('faces.db')
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def setup_gui(self):
        """Setup Tkinter GUI"""
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title label
        title = tk.Label(main_frame, text="Face Processing Application", 
                        font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 20))
        
        # Create buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))
        
        # Create buttons
        process_btn = tk.Button(button_frame, text="Process Unprocessed Images",
                              command=self.process_faces,
                              font=('Helvetica', 10))
        process_btn.pack(side=tk.LEFT, padx=5)
        
        recognize_btn = tk.Button(button_frame, text="Recognize Faces",
                               command=self.recognize_faces,
                               font=('Helvetica', 10))
        recognize_btn.pack(side=tk.LEFT, padx=5)
        
        quit_btn = tk.Button(button_frame, text="Quit",
                           command=self.root.quit,
                           font=('Helvetica', 10))
        quit_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create content frame for image and legend
        content_frame = tk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create image frame with fixed size
        self.image_frame = tk.Frame(content_frame, width=500, height=400)
        self.image_frame.pack(side=tk.LEFT, pady=20, padx=(0, 10))
        self.image_frame.pack_propagate(False)
        
        # Create image label
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True)
        
        # Create legend frame
        self.legend_frame = tk.Frame(content_frame, width=200)
        self.legend_frame.pack(side=tk.LEFT, fill=tk.Y, pady=20)
        
        # Create scrollable legend
        self.legend_canvas = tk.Canvas(self.legend_frame)
        scrollbar = tk.Scrollbar(self.legend_frame, orient="vertical", 
                               command=self.legend_canvas.yview)
        self.scrollable_legend = tk.Frame(self.legend_canvas)
        
        self.scrollable_legend.bind(
            "<Configure>",
            lambda e: self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
        )
        
        self.legend_canvas.create_window((0, 0), window=self.scrollable_legend, anchor="nw")
        self.legend_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create status label
        self.status_label = tk.Label(main_frame, text="Ready", 
                                   font=('Helvetica', 10))
        self.status_label.pack(pady=10)

    def is_image_processed(self, image_path):
        """Check if image has already been processed"""
        self.cursor.execute('SELECT * FROM processed_images WHERE original_path = ?', 
                          (image_path,))
        return self.cursor.fetchone() is not None

    def get_next_person_id(self):
        """Get the next available person ID"""
        self.cursor.execute('SELECT MAX(id) FROM persons')
        result = self.cursor.fetchone()[0]
        return 1 if result is None else result + 1

    def display_image(self, cv2_image, face_boxes=None):
        """Display image in the image frame with optional face boxes"""
        # Convert CV2 image to PIL format
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        
        # Resize image to fit frame while maintaining aspect ratio
        display_size = (480, 380)  # Slightly smaller than frame
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Draw face boxes if provided
        if face_boxes is not None:
            draw = ImageDraw.Draw(pil_image)
            # Scale boxes to resized image
            scale_x = pil_image.width / cv2_image.shape[1]
            scale_y = pil_image.height / cv2_image.shape[0]
            for box in face_boxes:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle(
                    [(x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y)],
                    outline='green', width=2
                )
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference

    def ask_name(self, parent, image_file):
        """Create a name input dialog"""
        dialog = tk.Toplevel(parent)
        dialog.title("Enter Name")
        dialog.geometry("400x200")
        
        # Make dialog modal
        dialog.transient(parent)
        dialog.grab_set()
        
        # Center the dialog on parent
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (400 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (200 // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Create and pack widgets
        tk.Label(dialog, text=f"Enter name for face in {image_file}").pack(pady=20)
        
        name_var = tk.StringVar()
        entry = tk.Entry(dialog, textvariable=name_var)
        entry.pack(pady=10)
        entry.focus()
        
        def submit():
            if name_var.get().strip():
                dialog.destroy()
        
        tk.Button(dialog, text="Save", command=submit).pack(pady=20)
        
        # Handle dialog close
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        dialog.wait_window()
        return name_var.get().strip()

    def process_faces(self):
        """Process all unprocessed images with enhanced UI feedback"""
        # Expanded list of supported image formats
        SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif')
        
        # Get list of image files with expanded formats
        image_files = [f for f in os.listdir(self.dirs['unprocessed']) 
                      if f.lower().endswith(SUPPORTED_FORMATS)]
        
        if not image_files:
            messagebox.showinfo("Info", 
                              f"No unprocessed images found in {self.dirs['unprocessed']}!\n"
                              f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            return
        
        processed_count = 0
        for image_file in image_files:
            image_path = os.path.join(self.dirs['unprocessed'], image_file)
            
            # Skip if already processed
            if self.is_image_processed(image_path):
                continue
            
            try:
                # Update status
                self.status_label.config(text=f"Processing {image_file}...")
                self.root.update()
                
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    raise Exception("Could not read image")
                
                # Run face detection
                results = self.model(img)
                
                # Process each detected face
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    
                    if len(boxes) == 0:
                        print(f"No faces detected in {image_file}")
                        continue
                    
                    # Process each face one by one
                    for i, box in enumerate(boxes):
                        # Create a copy of original image for display
                        display_img = img.copy()
                        
                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw rectangle only for the current face
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Display image with current face highlighted
                        self.display_image(display_img)
                        
                        # Ask for name using enhanced dialog
                        name = self.ask_name(self.root, f"face {i+1} in {image_file}")
                        if not name:
                            continue
                        
                        # Crop and save face
                        face_img = img[y1:y2, x1:x2]
                        face_filename = f"person_{self.get_next_person_id()}.jpg"
                        face_path = os.path.join(self.dirs['processed'], face_filename)
                        cv2.imwrite(face_path, face_img)
                        
                        # Save to database
                        self.cursor.execute(
                            'INSERT INTO persons (id, name, image_path) VALUES (?, ?, ?)',
                            (self.get_next_person_id(), name, face_path)
                        )
                        self.conn.commit()
                
                # Mark image as processed
                self.cursor.execute('INSERT INTO processed_images (original_path) VALUES (?)',
                                  (image_path,))
                self.conn.commit()
                processed_count += 1
                
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
                messagebox.showerror("Error", f"Error processing {image_file}: {str(e)}")
                continue
        
        if processed_count > 0:
            messagebox.showinfo("Success", 
                              f"Processing completed! Processed {processed_count} images.")
        else:
            messagebox.showinfo("Info", 
                              "No new images were processed. All images may have been processed already.")

    def update_legend(self, face_info):
        """Update the legend with face information"""
        # Clear previous legend entries
        for widget in self.scrollable_legend.winfo_children():
            widget.destroy()
        
        # Create new legend entries
        for name, confidence, color in face_info:
            # Create frame for each entry
            entry_frame = tk.Frame(self.scrollable_legend)
            entry_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Create color box
            color_box = tk.Canvas(entry_frame, width=20, height=20, 
                                highlightthickness=0)
            color_box.pack(side=tk.LEFT, padx=(0, 5))
            # Convert BGR to RGB for Tkinter
            rgb_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
            color_box.create_rectangle(0, 0, 20, 20, fill=rgb_color, outline='')
            
            # Create label with name and confidence
            tk.Label(entry_frame, 
                    text=f"{name} ({confidence:.2f})",
                    font=('Helvetica', 10)).pack(side=tk.LEFT)

    def recognize_faces(self):
        """Recognize faces in an image using face_recognition library"""
        from tkinter import filedialog
        import face_recognition
        
        # Open file picker
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not image_path:
            return
        
        try:
            # Update status
            self.status_label.config(text="Processing image...")
            self.root.update()
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", "Could not read image!")
                return
            
            # Detect faces using YOLO
            results = self.model(img)
            
            # Get face encodings from database
            known_face_encodings = []
            known_face_names = []
            
            # Query database for processed faces
            self.cursor.execute('SELECT id, name, image_path FROM persons')
            for person_id, name, face_path in self.cursor.fetchall():
                try:
                    known_img = face_recognition.load_image_file(face_path)
                    face_encoding = face_recognition.face_encodings(known_img)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(name)
                except Exception as e:
                    print(f"Error loading face {name}: {str(e)}")
            
            # Generate random colors for each detected face
            face_colors = []
            face_info = []  # List to store name and confidence for legend
            
            # Process each detected face
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                # Convert image to RGB for face_recognition
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process each detected face
                for box in boxes:
                    # Generate a random color for this face
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    face_colors.append(color)
                    
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract face location for face_recognition
                    face_location = (y1, x2, y2, x1)
                    
                    # Get face encoding
                    face_encoding = face_recognition.face_encodings(rgb_img, [face_location])[0]
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    best_match_index = -1
                    if len(face_distances) > 0:
                        best_match_index = int(np.argmin(face_distances))
                    
                    if best_match_index >= 0 and matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                    else:
                        name = "Unknown"
                        confidence = 0.0
                    
                    # Draw rectangle with the assigned color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Store face info for legend
                    face_info.append((name, confidence, color))
            
            # Display the result image
            self.display_image(img)
            
            # Update the legend
            self.update_legend(face_info)
            
            self.status_label.config(text="Recognition completed!")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    def run(self):
        """Start the application"""
        self.root.mainloop()
        self.conn.close()

if __name__ == "__main__":
    app = FaceProcessor()
    app.run()
