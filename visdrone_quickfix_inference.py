import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import json
import os


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
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

    def preprocess_image(self, image_path):
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((320, 320), Image.BILINEAR)
            img_array = np.array(img)
            return np.expand_dims(img_array, axis=0).astype(np.uint8)

    def post_process_predictions(self, boxes, classes, confidence_threshold=0.5):
        """Post-process predictions with confidence filtering and NMS"""
        # Get confidence scores and class IDs
        scores = np.max(classes, axis=-1)
        class_ids = np.argmax(classes, axis=-1)

        # Filter by confidence
        valid_detections = scores > confidence_threshold
        filtered_boxes = boxes[valid_detections]
        filtered_scores = scores[valid_detections]
        filtered_classes = class_ids[valid_detections]

        # Skip if no valid detections
        if len(filtered_boxes) == 0:
            return [], [], []

        # Convert boxes to [y1, x1, y2, x2] format for NMS
        nms_boxes = filtered_boxes

        # Apply NMS
        keep_indices = []
        for class_id in np.unique(filtered_classes):
            class_mask = filtered_classes == class_id
            class_boxes = nms_boxes[class_mask]
            class_scores = filtered_scores[class_mask]

            # Apply NMS per class
            indices = self._non_max_suppression(
                class_boxes, class_scores, iou_threshold=0.5, max_detections=20
            )

            # Add class indices to keep list
            class_indices = np.where(class_mask)[0][indices]
            keep_indices.extend(class_indices)

        # Sort by score
        keep_indices = sorted(
            keep_indices, key=lambda i: filtered_scores[i], reverse=True
        )

        return (
            filtered_boxes[keep_indices],
            filtered_scores[keep_indices],
            filtered_classes[keep_indices],
        )

    def _non_max_suppression(self, boxes, scores, iou_threshold=0.5, max_detections=20):
        """Custom NMS implementation"""
        if len(boxes) == 0:
            return []

        # Initialize the list of picked indexes
        pick = []

        # Calculate areas
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)

        # Sort by confidence score
        idxs = np.argsort(scores)

        # Keep looping while some indexes still remain
        while len(idxs) > 0:
            # Grab the last index and add it to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            if len(pick) >= max_detections:
                break

            # Find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes that have high overlap
            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0]))
            )

        return pick

    def run_inference(self, image_path):
        """Run inference with post-processing"""
        try:
            # Preprocess and run inference
            input_data = self.preprocess_image(image_path)
            start_time = time.perf_counter()

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            # Get raw outputs
            raw_boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
            raw_classes = self.interpreter.get_tensor(self.output_details[1]["index"])[
                0
            ]

            inference_time = time.perf_counter() - start_time

            # Post-process predictions
            boxes, scores, classes = self.post_process_predictions(
                raw_boxes,
                raw_classes,
                confidence_threshold=0.3,  # Adjust this threshold as needed
            )

            # Convert to list of detections
            detections = []
            for box, score, class_id in zip(boxes, scores, classes):
                if class_id == 0:  # Skip 'ignored regions'
                    continue
                detections.append(
                    {
                        "class": self.CATEGORY_MAPPING[class_id],
                        "confidence": float(score),
                        "box": box.tolist(),
                    }
                )

            return {"inference_time": inference_time, "detections": detections}

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            raise

    def process_directory(self, image_dir, results_path):
        """Process directory of images"""
        all_results = []
        total_time = 0
        num_images = 0

        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            try:
                results = self.run_inference(image_path)

                all_results.append(
                    {
                        "image": filename,
                        "inference_time_ms": results["inference_time"] * 1000,
                        "detections": results["detections"],
                    }
                )

                total_time += results["inference_time"]
                num_images += 1

                print(f"Processed {filename}: {len(results['detections'])} detections")

            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
                continue

        # Save results
        if num_images > 0:
            avg_time = (total_time / num_images) * 1000
            total_detections = sum(len(r["detections"]) for r in all_results)

            summary = {
                "total_images": len(image_files),
                "processed_images": num_images,
                "average_inference_time_ms": avg_time,
                "total_detections": total_detections,
                "detections_per_image": total_detections / num_images,
                "detailed_results": all_results,
            }

            with open(results_path, "w") as f:
                json.dump(summary, f, indent=2)

            print("\nProcessing complete!")
            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Results saved to: {results_path}")


def main():
    MODEL_PATH = "visdrone_model_edge_tpu.tflite"
    IMAGE_DIR = "test_images/VisDrone-test-challenge-smol-dataset"
    results_path = "inference_results/inference_results.json"

    detector = VisDroneInference(MODEL_PATH)
    detector.process_directory(IMAGE_DIR, results_path)


if __name__ == "__main__":
    main()
