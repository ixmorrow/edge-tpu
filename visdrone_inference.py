import os
import json
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from datetime import datetime


class VisDroneInference:
    CATEGORY_MAPPING = {
        0: "ignored regions",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
        11: "others",
    }

    def __init__(self, model_path):
        # Initialize TFLite interpreter with Edge TPU
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input size
        self.input_shape = self.input_details[0]["shape"]
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load and resize image
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((self.width, self.height))

        # Convert to numpy array
        image = np.array(image, dtype=np.uint8)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image

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
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])
        labels = self.interpreter.get_tensor(self.output_details[1]["index"])

        inference_time = time.perf_counter() - start_time

        return {"boxes": boxes, "labels": labels, "inference_time": inference_time}

    def process_directory(self, image_dir, results_path):
        """Process all images in a directory"""
        all_results = []
        total_time = 0
        num_images = 0

        # Process each image in directory
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(image_dir, filename)
                try:
                    results = self.run_inference(image_path)

                    # Filter confident detections
                    confident_detections = []
                    for box, cls_probs in zip(results["boxes"], results["labels"]):
                        confidence = np.max(cls_probs)
                        if confidence > 0.5:
                            class_id = np.argmax(cls_probs)
                            confident_detections.append(
                                {
                                    "class": self.CATEGORY_MAPPING[class_id],
                                    "confidence": float(confidence),
                                    "box": box.tolist(),
                                }
                            )

                    # Record results
                    all_results.append(
                        {
                            "image": filename,
                            "inference_time_ms": results["inference_time"] * 1000,
                            "detections": confident_detections,
                        }
                    )

                    total_time += results["inference_time"]
                    num_images += 1

                    print(
                        f"Processed {filename}: {len(confident_detections)} detections"
                    )

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        # Calculate summary statistics
        if num_images > 0:
            avg_time = (total_time / num_images) * 1000  # Convert to ms
            total_detections = sum(len(r["detections"]) for r in all_results)

            summary = {
                "total_images": num_images,
                "average_inference_time_ms": avg_time,
                "total_detections": total_detections,
                "detections_per_image": total_detections / num_images,
                "detailed_results": all_results,
            }

            # Save results
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)

            print("\nProcessing complete!")
            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Total detections: {total_detections}")
            print(f"Results saved to: {results_path}")
        else:
            print("No images processed!")

    def save_results(self, results, image_path, output_dir):
        """Save inference results to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_image_name = os.path.basename(image_path)
        output_path = os.path.join(
            output_dir, f"inference_results_{base_image_name}_{timestamp}.txt"
        )

        # Get original image dimensions for scaling boxes
        original_image = Image.open(image_path)
        orig_width, orig_height = original_image.size

        with open(output_path, "w") as f:
            f.write(f"Results for image: {base_image_name}\n")
            f.write(f"Inference time: {results['inference_time']*1000:.2f}ms\n\n")

            boxes = results["boxes"][0]  # First batch
            labels = results["labels"][0]  # First batch

            for i in range(len(boxes)):
                # Skip if box coordinates are all zero
                if not np.any(boxes[i]):
                    continue

                # Get class label
                class_id = np.argmax(labels[i])
                confidence = labels[i][class_id]

                # Only include detections with reasonable confidence
                if confidence > 0.5:
                    # Scale boxes to original image dimensions
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin = xmin * orig_width
                    xmax = xmax * orig_width
                    ymin = ymin * orig_height
                    ymax = ymax * orig_height

                    f.write(f"Detection {i+1}:\n")
                    f.write(
                        f"  Class: {self.CATEGORY_MAPPING[class_id]} (ID: {class_id})\n"
                    )
                    f.write(f"  Confidence: {confidence:.2f}\n")
                    f.write(f"  Box: [xmin={xmin:.1f}, ymin={ymin:.1f}, ")
                    f.write(f"xmax={xmax:.1f}, ymax={ymax:.1f}]\n\n")


def main():
    # Configuration
    MODEL_PATH = "visdrone_model_edge_tpu.tflite"
    IMAGE_DIR = "test_images/VisDrone-test-challenge-sample-dataset"
    OUTPUT_DIR = "inference_results"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize inference
    detector = VisDroneInference(MODEL_PATH)

    # # Process all images in directory
    # for image_file in os.listdir(IMAGE_DIR):
    #     if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    #         image_path = os.path.join(IMAGE_DIR, image_file)
    #         try:
    #             print(f"Processing {image_file}...")
    #             results = detector.run_inference(image_path)
    #             detector.save_results(results, image_path, OUTPUT_DIR)
    #             print(f"  Inference time: {results['inference_time']*1000:.2f}ms")
    #         except Exception as e:
    #             print(f"Error processing {image_file}: {str(e)}")
    detector.process_directory(IMAGE_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
