import requests
import os
import sys
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure it's running.")
        return False

def test_get_persons():
    """Test getting all persons"""
    response = requests.get(f"{BASE_URL}/persons")
    if response.status_code == 200:
        persons = response.json()
        print(f"✅ Got {len(persons)} persons from the database")
        return persons
    else:
        print(f"❌ Failed to get persons: {response.status_code}")
        return []

def test_add_person(name, image_path):
    """Test adding a new person"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'face_image': f}
        data = {'name': name}
        response = requests.post(f"{BASE_URL}/persons", files=files, data=data)
    
    if response.status_code == 200:
        person = response.json()
        print(f"✅ Added person: {person['name']} (ID: {person['id']})")
        return person
    else:
        print(f"❌ Failed to add person: {response.status_code}")
        print(response.text)
        return None

def test_update_person(person_id, new_name):
    """Test updating a person"""
    data = {'name': new_name}
    response = requests.put(f"{BASE_URL}/persons/{person_id}", json=data)
    
    if response.status_code == 200:
        person = response.json()
        print(f"✅ Updated person {person_id} to name: {person['name']}")
        return person
    else:
        print(f"❌ Failed to update person: {response.status_code}")
        print(response.text)
        return None

def test_delete_person(person_id):
    """Test deleting a person"""
    response = requests.delete(f"{BASE_URL}/persons/{person_id}")
    
    if response.status_code == 200:
        print(f"✅ Deleted person with ID: {person_id}")
        return True
    else:
        print(f"❌ Failed to delete person: {response.status_code}")
        print(response.text)
        return False

def test_detect_faces(image_path):
    """Test face detection"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/detect-faces", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Detected {len(result['faces'])} faces in the image")
        return result
    else:
        print(f"❌ Failed to detect faces: {response.status_code}")
        print(response.text)
        return None

def test_recognize_faces(image_path):
    """Test face recognition"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/recognize", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Recognized {len(result['faces'])} faces in the image")
        for i, face in enumerate(result['faces']):
            print(f"  Face {i+1}: {face['name']} (Confidence: {face['confidence']:.2f})")
        return result
    else:
        print(f"❌ Failed to recognize faces: {response.status_code}")
        print(response.text)
        return None

def main():
    """Run all tests"""
    print("Testing Face Recognition API...")
    
    # Check if API is running
    if not test_api_health():
        print("Exiting tests due to API connection failure.")
        sys.exit(1)
    
    # Get all persons
    persons = test_get_persons()
    
    # If command line arguments are provided, use them for testing
    if len(sys.argv) > 2:
        action = sys.argv[1]
        
        if action == "add" and len(sys.argv) > 3:
            # Add a person
            name = sys.argv[2]
            image_path = sys.argv[3]
            test_add_person(name, image_path)
        
        elif action == "update" and len(sys.argv) > 3:
            # Update a person
            person_id = int(sys.argv[2])
            new_name = sys.argv[3]
            test_update_person(person_id, new_name)
        
        elif action == "delete" and len(sys.argv) > 2:
            # Delete a person
            person_id = int(sys.argv[2])
            test_delete_person(person_id)
        
        elif action == "detect" and len(sys.argv) > 2:
            # Detect faces
            image_path = sys.argv[2]
            test_detect_faces(image_path)
        
        elif action == "recognize" and len(sys.argv) > 2:
            # Recognize faces
            image_path = sys.argv[2]
            test_recognize_faces(image_path)
        
        else:
            print("Invalid action or missing arguments.")
            print("Usage:")
            print("  python test_api.py add <name> <image_path>")
            print("  python test_api.py update <person_id> <new_name>")
            print("  python test_api.py delete <person_id>")
            print("  python test_api.py detect <image_path>")
            print("  python test_api.py recognize <image_path>")
    else:
        print("\nUsage:")
        print("  python test_api.py add <name> <image_path>")
        print("  python test_api.py update <person_id> <new_name>")
        print("  python test_api.py delete <person_id>")
        print("  python test_api.py detect <image_path>")
        print("  python test_api.py recognize <image_path>")

if __name__ == "__main__":
    main() 