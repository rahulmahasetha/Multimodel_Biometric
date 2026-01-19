import os

def get_all_persons():
    """Get sorted list of all persons from any modality"""
    persons = []
    
    # Try to get from face directory
    face_path = "../data/face"
    if os.path.exists(face_path):
        persons = [p for p in os.listdir(face_path) 
                  if os.path.isdir(os.path.join(face_path, p))]
    
    # If no face data, try other modalities
    if not persons:
        for modality in ["finger", "palm"]:
            modality_path = f"../data/{modality}"
            if os.path.exists(modality_path):
                persons = [p for p in os.listdir(modality_path) 
                          if os.path.isdir(os.path.join(modality_path, p))]
                if persons:
                    break
    
    persons.sort()  # Consistent ordering
    print(f"Found {len(persons)} persons: {persons}")
    return persons