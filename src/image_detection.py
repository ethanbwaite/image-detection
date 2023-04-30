import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection, DetrImageProcessor, DetrForObjectDetection, AutoFeatureExtractor, AutoModelForObjectDetection
import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU for inference...')
else:
    device = torch.device('cpu')
    print('Using CPU for inference...')


# Load the YOLOv5 object detection model from Hugging Face
print("Loading models...")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-101")

# Move the models to GPU
# feature_extractor.to(device)
model.to(device)

config = model.config
labels_dict = config.id2label

def interpolate_color(confidence, min_confidence=0.0, max_confidence=1.0):
    # Map confidence score to a value between 0 and 1
    normalized_confidence = max(0, min(1, (confidence - min_confidence) / (max_confidence - min_confidence)))

    # Calculate the R, G, and B values of the color
    r = int(255 * (1 - normalized_confidence))
    g = int(255 * normalized_confidence)
    b = 0

    return (b, g, r)

# Define the function to detect objects in an image using the YOLOv5 model
def detect_objects(image, should_detect_objects=True):
    image_width, image_height, image_channels = image.shape
    resized_width, resized_height = 1280, 720
    confidence_threshold = 0.5
    scale_factor_x = image_width / resized_width
    scale_factor_y = image_height / resized_height

    resized_image = cv2.resize(image, (resized_width, resized_height))

    if should_detect_objects:
        # Convert the image to RGB format and pass it through the feature extractor
        inputs = feature_extractor(images=resized_image[:, :, ::-1], return_tensors="pt", image_size=(resized_width, resized_height)).to(device)

        # Use the object detection model to make predictions on the input image
        outputs = model(**inputs)
        results = feature_extractor.post_process_object_detection(outputs, threshold=confidence_threshold)[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_rounded = [round(i, 2) for i in box.tolist()]
            label_text = model.config.id2label[label.item()]
            print(
                f"Detected {label_text} with confidence "
                f"{round(score.item(), 3)} at location {box_rounded}"
            )

            x1, y1, x2, y2 = box
            x1, y1 = int(x1 * resized_width), int(y1 * resized_height)
            x2, y2 = int(x2 * resized_width), int(y2 * resized_height)

            color = interpolate_color(score.item(), min_confidence=confidence_threshold)
            thickness = 2

            cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(resized_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return resized_image
