import random
import string
import util
import numpy as np
import cv2
import colorsys
from constants import KNOWN_OBJECT_AGE, KNOWN_OBJECT_TIME_SINCE_LAST_DETECTION, KNOWN_OBJECT_HISTORY, KNOWN_OBJECT_COLOR, KNOWN_OBJECT_LABEL

log = util.get_logger()

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


def get_centroid(box):
    """
    Returns the centroid of a bounding box given its coordinates.
    
    Parameters:
        box (list): A list of four integers representing the bounding box coordinates in the format [xmin, ymin, xmax, ymax].
        
    Returns:
        centroid (tuple): A tuple of two floating-point values representing the x and y coordinates of the centroid.
    """
    xmin, ymin, xmax, ymax = box
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    return (x, y)


def generate_color():
    """
    Generates a random bright and happy color (no browns or muddy tones)
    Returns: tuple of (b, g, r) color values
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def generate_saturated_color():
    """
    Generates a random, highly saturated (b, g, r) color in OpenCV.
    """
    hsv = np.array([[[np.random.randint(0, 180), 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def fade_to_white(color, percent):
    """Fades a color to white by removing saturation.
    
    Args:
        color (tuple): A tuple containing (b, g, r) values.
        percent (float): A value between 0 and 1 indicating how much to fade the color.
        
    Returns:
        tuple: A tuple containing the faded (b, g, r) values.
    """
    # Convert color to hsv
    hsv_color = colorsys.rgb_to_hsv(*color)
    
    # Fade saturation to 0 over percent of the range
    faded_hsv_color = (hsv_color[0], hsv_color[1] * (1 - percent), hsv_color[2])
    
    # Convert faded color back to (b, g, r)
    faded_color = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*faded_hsv_color))
    
    return faded_color


# Maintain a state of known objects based on the object bounding box data of the current frame.
def track_object(known_objects, 
                 known_object_metadata, 
                 candidate_objects, 
                 overlap_threshold=0.5, 
                 size_delta_threshold=0.5, 
                 time_since_last_detection_threshold=2,
                 merge_overlap_threshold=0.95,
                 merge_size_delta_threshold=0.05):
    new_known_objects = known_objects.copy()
    
    # Keep track of object IDs that have continued to exist in this frame
    # Any that we did not find a candidate for we will remove at the end
    continuous_object_ids = set()

    for candidate_label, candidate_bbox in candidate_objects:

        # Track if we found a new bbox for an existing object.
        # Otherwise we lost tracking and should remove it
        found_object = False
        for known_id, known_bbox in known_objects.items():
            overlap = calculate_bbox_overlap(known_bbox, candidate_bbox)
            size_delta = calculate_size_delta(known_bbox, candidate_bbox)
            is_same_class = known_object_metadata[known_id][KNOWN_OBJECT_LABEL] == candidate_label
            
            # Known object detected
            # Assign known object to new location if we can reasonably say it is the same object
            if overlap > overlap_threshold and size_delta < size_delta_threshold:
                # Continue tracking the object
                continuous_object_ids.add(known_id)
                new_known_objects[known_id] = candidate_bbox

                # Update this known object's metadata
                known_object_metadata[known_id][KNOWN_OBJECT_AGE] += 1
                known_object_metadata[known_id][KNOWN_OBJECT_TIME_SINCE_LAST_DETECTION] = 0
                history = known_object_metadata[known_id][KNOWN_OBJECT_HISTORY].copy()
                known_object_metadata[known_id][KNOWN_OBJECT_HISTORY] = util.add_history(
                    history, 
                    candidate_bbox,
                    max_history=100
                )
                found_object = True
        
        # This is a new known object, add it to the set
        if not found_object:
            new_id = f"{candidate_label}-{generate_random_id()}"
            new_known_objects[new_id] = candidate_bbox
            known_object_metadata[new_id] = {}
            known_object_metadata[new_id][KNOWN_OBJECT_LABEL] = candidate_label
            known_object_metadata[new_id][KNOWN_OBJECT_AGE] = 0
            known_object_metadata[new_id][KNOWN_OBJECT_TIME_SINCE_LAST_DETECTION] = 0
            known_object_metadata[new_id][KNOWN_OBJECT_HISTORY] = []
            known_object_metadata[new_id][KNOWN_OBJECT_COLOR] = generate_saturated_color()

    # Objects without a new mapping are candidates for removal from the state
    non_continuous_objects = known_objects.keys() - continuous_object_ids
    non_continuous_objects_in_grace_period = set()

    # Merge objects that are clearly too similar and likely tracking the same thing
    # In these cases, keep the older object
    merged_removed_objects = set()

    for known_id_1, known_bbox_1 in known_objects.items():
        for known_id_2, known_bbox_2 in known_objects.items():
            if known_id_1 != known_id_2:
                overlap = calculate_bbox_overlap(known_bbox_1, known_bbox_2)
                size_delta = calculate_size_delta(known_bbox_1, known_bbox_2)

                if overlap > merge_overlap_threshold and size_delta < merge_size_delta_threshold:
                    if known_object_metadata[known_id_1][KNOWN_OBJECT_AGE] > known_object_metadata[known_id_2][KNOWN_OBJECT_AGE]:
                        merged_removed_objects.add(known_id_2)
                        # log.info(f"Merged {known_id_2} into {known_id_1}")
                    else:
                        merged_removed_objects.add(known_id_1)
                        # log.info(f"Merged {known_id_1} into {known_id_2}")


    # Allow objects that have not been detected to persist so long as they are within a time threshold
    for object_id in non_continuous_objects:
        if known_object_metadata[object_id][KNOWN_OBJECT_TIME_SINCE_LAST_DETECTION] < time_since_last_detection_threshold:
            known_object_metadata[object_id][KNOWN_OBJECT_TIME_SINCE_LAST_DETECTION] += 1
            known_object_metadata[object_id][KNOWN_OBJECT_AGE] += 1
            non_continuous_objects_in_grace_period.add(object_id)

    non_continuous_objects = non_continuous_objects - non_continuous_objects_in_grace_period
    non_continuous_objects = non_continuous_objects.union(merged_removed_objects)
    # Return the new set of known objects, minus any that we did not find a candidate for 
    # from the new frame 
    final_known_objects = {k: v for k, v in new_known_objects.items() if k not in non_continuous_objects}
    final_known_object_metadata = {k: v for k, v in known_object_metadata.items() if k not in non_continuous_objects}

    return final_known_objects, final_known_object_metadata