import random
import string


def generate_random_id():
    # Generate a random 4-digit ID
    return ''.join(random.choices(string.digits, k=4))


def calculate_bbox_overlap(old_bbox, new_bbox):
    """
    Calculate the percentage area overlap between two bounding boxes.

    Args:
        bbox1 (tuple): The first bounding box in the format of (x1, y1, x2, y2).
        bbox2 (tuple): The second bounding box in the format of (x1, y1, x2, y2).

    Returns:
        float: The percentage area overlap between the two bounding boxes.
    """
    old_xmin, old_ymin, old_xmax, old_ymax = old_bbox
    new_xmin, new_ymin, new_xmax, new_ymax = new_bbox
    
    # Calculate the area of the old and new bounding boxes
    old_area = (old_xmax - old_xmin) * (old_ymax - old_ymin)
    new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)
    
    # Calculate the coordinates of the intersection between the old and new bounding boxes
    xmin = max(old_xmin, new_xmin)
    ymin = max(old_ymin, new_ymin)
    xmax = min(old_xmax, new_xmax)
    ymax = min(old_ymax, new_ymax)
    
    # Calculate the area of the intersection
    intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    
    # Calculate the percentage of overlap
    overlap_percentage = intersection_area / min(old_area, new_area) * 100
    
    return overlap_percentage


def calculate_size_delta(bbox1, bbox2):
    """
    Calculate the size delta between two bounding boxes.

    Args:
        bbox1 (tuple): The first bounding box in the format of (x1, y1, x2, y2).
        bbox2 (tuple): The second bounding box in the format of (x1, y1, x2, y2).

    Returns:
        float: The size delta between the two bounding boxes.
    """
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    return abs(area2 - area1) / area1


def track_object(known_objects, known_object_metadata, candidate_objects, overlap_threshold=0.5, size_delta_threshold=0.5):
    new_known_objects = known_objects.copy()
    
    # Keep track of object IDs that have continued to exist in this frame
    # Any that we did not find a candidate for we will remove 
    continuous_object_ids = set()

    for candidate_bbox in candidate_objects:
        print(f"Candidate Objects: {len(candidate_objects)}")
        print(f"Knonw Objects: {len(known_objects.items())}")

        # Track if we found a new bbox for an existing object.
        # Otherwise we lost tracking and should remove it
        found_object = False
        for known_id, known_bbox in known_objects.items():
            overlap = calculate_bbox_overlap(known_bbox, candidate_bbox)
            size_delta = calculate_size_delta(known_bbox, candidate_bbox)

            # Assign known object to new location if we can reasonably say it is the same object
            if overlap > overlap_threshold and size_delta < size_delta_threshold:
                continuous_object_ids.add(known_id)
                new_known_objects[known_id] = candidate_bbox
                found_object = True
        
        if not found_object:
            # This is a new known object, add it to the set
            new_known_objects[generate_random_id()] = candidate_bbox

    non_continuous_objects = known_objects.keys() - continuous_object_ids

    # Return the new set of known objects, minus any that we did not find a candidate for 
    # from the new frame 
    return {k: v for k, v in new_known_objects.items() if k not in non_continuous_objects}