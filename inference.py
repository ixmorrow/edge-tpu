import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2
import time
from ultralytics import YOLO


class CoralYOLODetector:
    def __init__(self, model_path, labels_path, conf_threshold=0.3):
        """
        Initialize the YOLO detector for Coral TPU.

        Args:
            model_path: Path to the converted TFLite model
            labels_path: Path to the labels file
            conf_threshold: Confidence threshold for detections
        """
        # Load labels
        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Initialize the TF Lite interpreter
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]
        self.conf_threshold = conf_threshold

    def preprocess_image(self, image_path):
        """
        Preprocess the image to match model input requirements.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image array
        """
        # Load and resize image
        image = Image.open(image_path)
        image = image.resize((self.input_shape[1], self.input_shape[2]))

        # Convert to array and normalize
        image_array = np.array(image)
        image_array = image_array.astype(np.float32)
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def detect(self, image_path):
        """
        Perform object detection on the input image.

        Args:
            image_path: Path to input image

        Returns:
            List of detections, each containing class_id, class_name, confidence, and bounding box
        """
        # Preprocess image
        input_data = self.preprocess_image(image_path)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output tensors
        # Assuming output format: [batch, num_detections, 6] where 6 is [x, y, w, h, confidence, class]
        detections = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Process detections
        results = []
        for detection in detections[0]:  # Process first batch only
            confidence = detection[4]

            if confidence > self.conf_threshold:
                class_id = int(detection[5])
                class_name = self.labels[class_id]
                bbox = detection[:4]  # [x, y, width, height]

                results.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": bbox.tolist(),
                    }
                )

        return results, inference_time


def visualize_detections(image_path, detections):
    """
    Visualize the detected objects on the image.

    Args:
        image_path: Path to original image
        detections: List of detections from the detect() method
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for det in detections:
        bbox = det["bbox"]
        x, y, w, h = bbox

        # Convert normalized coordinates to pixel coordinates
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image


# Example usage
def main():
    # Initialize detector
    detector = CoralYOLODetector(
        model_path="path/to/your/model.tflite",
        labels_path="path/to/your/labels.txt",
        conf_threshold=0.3,
    )

    # Perform detection
    image_path = "path/to/your/image.jpg"
    detections, inference_time = detector.detect(image_path)

    # Print results
    print(f"Inference time: {inference_time:.2f} seconds")
    for det in detections:
        print(f"Detected {det['class_name']} with confidence {det['confidence']:.2f}")

    # Visualize results
    result_image = visualize_detections(image_path, detections)
    cv2.imshow("Detections", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
