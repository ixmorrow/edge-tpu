import os
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from datetime import datetime


class CoralInference:
    def __init__(self, model_path, label_path):
        # Load labels
        with open(label_path, "r") as f:
            self.labels = {i: line.strip() for i, line in enumerate(f.readlines())}

        # Initialize TFLite interpreter with Edge TPU
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Resize image to model input size
        image = image.resize((self.width, self.height))
        # Convert to numpy array
        image = np.array(image)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image.astype(np.uint8)  # Note: using uint8 for quantized model

    def run_inference(self, image_path):
        """Run inference on a single image"""
        # Time the inference
        start_time = time.perf_counter()

        # Preprocess image
        input_data = self.preprocess_image(image_path)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensors
        # Note: Output tensor indices are different for the pre-compiled model
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        count = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        inference_time = time.perf_counter() - start_time

        # Count only high-confidence detections
        confident_detections = sum(1 for score in scores if score > 0.5)

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "total_possible_detections": count,
            "confident_detections": confident_detections,
            "inference_time": inference_time,
        }

    def save_results(self, results, image_path, output_dir):
        """Save inference results to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_image_name = os.path.basename(image_path)
        output_path = os.path.join(
            output_dir, f"inference_results_{base_image_name}_{timestamp}.txt"
        )

        with open(output_path, "w") as f:
            f.write(f"Results for image: {base_image_name}\n")
            f.write(f"Inference time: {results['inference_time']*1000:.2f}ms\n")
            f.write(f"High confidence detections: {results['confident_detections']}\n")
            f.write(
                f"(Out of {results['total_possible_detections']} possible detection slots)\n\n"
            )

            for i in range(results["count"]):
                if results["scores"][i] > 0.5:  # Filter low confidence detections
                    class_id = int(results["classes"][i])
                    f.write(f"Detection {i+1}:\n")
                    f.write(f"  Class: {self.labels[class_id]} (ID: {class_id})\n")
                    f.write(f"  Confidence: {results['scores'][i]:.2f}\n")
                    f.write(f"  Box: [ymin={results['boxes'][i][0]:.2f}, ")
                    f.write(f"xmin={results['boxes'][i][1]:.2f}, ")
                    f.write(f"ymax={results['boxes'][i][2]:.2f}, ")
                    f.write(f"xmax={results['boxes'][i][3]:.2f}]\n\n")


def main():
    # Configuration
    MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    LABEL_PATH = "coco_labels.txt"
    IMAGE_DIR = "test_images"
    OUTPUT_DIR = "output"

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize inference
    try:
        coral_detector = CoralInference(MODEL_PATH, LABEL_PATH)
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return

    # Process all images in directory
    total_images = 0
    successful_images = 0

    for image_file in os.listdir(IMAGE_DIR):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            total_images += 1
            image_path = os.path.join(IMAGE_DIR, image_file)
            try:
                print(f"Processing {image_file}...")
                results = coral_detector.run_inference(image_path)
                coral_detector.save_results(results, image_file, OUTPUT_DIR)
                print(f"  Found {results['count']} objects")
                print(f"  Inference time: {results['inference_time']*1000:.2f}ms")
                successful_images += 1
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    print(
        f"\nProcessing complete! Successfully processed {successful_images}/{total_images} images"
    )


if __name__ == "__main__":
    main()
