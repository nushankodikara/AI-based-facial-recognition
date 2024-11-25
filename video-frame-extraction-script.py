import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    print(f"Total frames: {frame_count}")
    print(f"FPS: {fps}")
    
    # Read and save frames
    frame_number = 0
    while True:
        # Read next frame
        success, frame = video.read()
        
        # Break if no more frames
        if not success:
            break
        
        # Save frame as image
        frame_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_number += 1
        
        # Print progress
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")
    
    # Release video object
    video.release()
    print(f"Extraction complete! {frame_number} frames saved to {output_folder}")

if __name__ == "__main__":
    # Example usage
    video_path = "/Users/nushankodikara/Desktop/rec.mov"  # Replace with your video path
    output_folder = "extracted_frames"      # Replace with your desired output folder
    
    extract_frames(video_path, output_folder)
