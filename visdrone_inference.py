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
        """Preprocess image with improved error handling"""
        try:
            with Image.open(image_path) as img:
                # Convert grayscale to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize image
                img = img.resize((320, 320), Image.Resampling.BILINEAR)

                # Convert to numpy array
                img_array = np.array(img)

                # Add batch dimension and ensure uint8 type
                input_data = np.expand_dims(img_array, axis=0).astype(np.uint8)

                return input_data

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            print("Image details:")
            try:
                with Image.open(image_path) as img:
                    print(f"  Mode: {img.mode}")
                    print(f"  Size: {img.size}")
                    print(f"  Format: {img.format}")
            except Exception as inner_e:
                print(f"  Unable to read image details: {str(inner_e)}")
            raise

    def run_inference(self, image_path):
        """Run inference with detailed error reporting"""
        try:
            # Preprocess image
            input_data = self.preprocess_image(image_path)

            # Time the inference
            start_time = time.perf_counter()

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get inference time
            inference_time = time.perf_counter() - start_time

            # Get outputs
            boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]

            return {
                "boxes": boxes,
                "classes": classes,
                "inference_time": inference_time,
            }

        except Exception as e:
            print(f"Detailed error for {image_path}:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            if hasattr(e, "errno"):
                print(f"  Error number: {e.errno}")
            raise

    def process_directory(self, image_dir, results_path):
        """Process all images in directory with error tracking"""
        all_results = []
        total_time = 0
        num_images = 0
        failed_images = []

        # Get total number of images
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        total_images = len(image_files)

        print(f"Starting processing of {total_images} images...")

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            try:
                # Run inference
                results = self.run_inference(image_path)

                # Filter confident detections
                confident_detections = []
                for box, cls_probs in zip(results["boxes"], results["classes"]):
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

                # Print progress
                print(
                    f"Processed {num_images}/{total_images}: {filename} - "
                    f"Found {len(confident_detections)} objects in "
                    f"{results['inference_time']*1000:.1f}ms"
                )

            except Exception as e:
                failed_images.append(
                    {"image": filename, "error": str(e), "error_type": type(e).__name__}
                )
                print(f"Failed to process {filename}: {str(e)}")
                continue

        # Calculate statistics
        if num_images > 0:
            avg_time = (total_time / num_images) * 1000
            total_detections = sum(len(r["detections"]) for r in all_results)

            summary = {
                "total_images": total_images,
                "processed_images": num_images,
                "failed_images": len(failed_images),
                "average_inference_time_ms": avg_time,
                "total_detections": total_detections,
                "detections_per_image": total_detections / num_images,
                "failed_images_list": failed_images,
                "detailed_results": all_results,
            }

            # Save results
            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)

            print("\nProcessing complete!")
            print(f"Successfully processed: {num_images}/{total_images} images")
            print(f"Failed to process: {len(failed_images)} images")
            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Total detections: {total_detections}")
            print(f"Results saved to: {results_path}")

            # Save failed images list separately
            if failed_images:
                failed_images_path = results_path.replace(
                    ".json", "_failed_images.json"
                )
                with open(failed_images_path, "w") as f:
                    json.dump(failed_images, f, indent=2)
                print(f"Failed images list saved to: {failed_images_path}")

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
    OUTPUT_FILE = "inference_results.json"
    results_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

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
    detector.process_directory(IMAGE_DIR, results_path)


if __name__ == "__main__":
    main()
