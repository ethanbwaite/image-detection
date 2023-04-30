import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection, DetrImageProcessor, DetrForObjectDetection, AutoFeatureExtractor, AutoModelForObjectDetection
import torch
import util

log = util.get_logger()
CONFIDENCE_THRESHOLD = 0.5

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    log.info('Using GPU for inference...')
else:
    device = torch.device('cpu')
    log.info('Using CPU for inference...')


# Load the YOLOv5 object detection model from Hugging Face
log.info("Loading models...")
feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Move the models to GPU
# feature_extractor.to(device)
model.to(device)
config = model.config
labels_dict = config.id2label
labels_allowlist = {'person', 'car', 'motorcycle', 'airplane', 'bus', 'truck'}

labels_draw_rectangle_allowlist = {'person', 'car', 'motorcycle', 'airplane', 'bus', 'truck'}


def interpolate_color(confidence, min_confidence=0.0, max_confidence=1.0):
    # Map confidence score to a value between 0 and 1
    normalized_confidence = max(0, min(1, (confidence - min_confidence) / (max_confidence - min_confidence)))

    # Calculate the R, G, and B values of the color
    r = int(255 * (1 - normalized_confidence))
    g = int(255 * normalized_confidence)
    b = 0

    return (b, g, r)


def rgb_to_gray(color):
    # Map confidence score to a value between 0 and 1
    (b, g, r) = color
    max_hue = max(b, g, r)

    return (max_hue, max_hue, max_hue)


# Define the function to detect objects in an image using the YOLOv5 model
def detect_objects(image, should_detect_objects=True, should_log_detections=False):
    image_width, image_height, image_channels = image.shape
    resized_width, resized_height = 1280, 720
    
    scale_factor_x = image_width / resized_width
    scale_factor_y = image_height / resized_height

    resized_image = cv2.resize(image, (resized_width, resized_height))
    results = {}

    if should_detect_objects:
        # Convert the image to RGB format and pass it through the feature extractor
        inputs = feature_extractor(images=resized_image[:, :, ::-1], return_tensors="pt", image_size=(resized_width, resized_height)).to(device)

        # Use the object detection model to make predictions on the input image
        outputs = model(**inputs)
        results = feature_extractor.post_process_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD)[0]

        scores = []
        labels = []
        boxes = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_rounded = [round(i, 2) for i in box.tolist()]
            label_text = model.config.id2label[label.item()]

            if label_text in labels_allowlist:
                scores.append(score.item())
                labels.append(label_text)
                boxes.append(box.tolist())
            if should_log_detections:
                print(
                    f"Detected {label_text} with confidence "
                    f"{round(score.item(), 3)} at location {box_rounded}"
                )
            
        results = {
            "scores": scores,
            "labels": labels, 
            "boxes": boxes
        }


    return resized_image, results

# Draw detected objects to a given frame based on metadata about the detected objects from the model
def draw_detected_objects(frame, results, greyscale=False):
    if len(results) > 0:
        width, height, _ = frame.shape

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

            [x1, y1, x2, y2] = box
            x1, y1 = int(x1 * height), int(y1 * width)
            x2, y2 = int(x2 * height), int(y2 * width)

            color = interpolate_color(score, min_confidence=CONFIDENCE_THRESHOLD)
            thickness = 2
            font_thickness = 1

            if greyscale:
                color = (255, 255, 255) # rgb_to_gray(color)
                thickness = 1

            if label in labels_draw_rectangle_allowlist:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if not greyscale:
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, font_thickness)
        
    return frame


# Draw detected objects to a given frame based on metadata about the detected objects from the model
def draw_known_objects(frame, known_objects):
    width, height, _ = frame.shape
    for object_id, object_bbox in known_objects.items():

        [x1, y1, x2, y2] = object_bbox
        x1, y1 = int(x1 * height), int(y1 * width)
        x2, y2 = int(x2 * height), int(y2 * width)

        color = (0, 255, 0)
        thickness = 2
        font_thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, object_id, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, font_thickness)
        

    return frame