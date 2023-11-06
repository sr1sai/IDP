import cv2
import os

# Path to the folder containing the images
input_folder = './phase_3/input'

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with your YOLO weights and config files
classes = []
with open("coco.names", "r") as f:  # Replace with your COCO names file
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection on an image
def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    objects_detected = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for confidence
                object_name = classes[class_id]
                objects_detected.append(object_name)

    return objects_detected

# Process images in the folder
detected_objects = []
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        objects = detect_objects(image_path)
        detected_objects.extend(objects)

# Show the list of detected objects
print("Detected objects in the images:")
print(detected_objects)
