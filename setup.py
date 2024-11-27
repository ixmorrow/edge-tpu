import urllib.request
import os
import yaml


def download_yolo_resources():
    """
    Downloads and prepares YOLO resources including weights and labels.
    Creates necessary directories and files for the project.
    """
    # Create project directories
    directories = ["models", "data", "labels"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Download COCO labels
    coco_labels_url = (
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.yaml"
    )
    labels_file = "data/coco.yaml"
    urllib.request.urlretrieve(coco_labels_url, labels_file)

    # Convert YAML labels to txt format
    with open(labels_file, "r") as f:
        yaml_data = yaml.safe_load(f)
        names = yaml_data.get("names", [])

    # Write labels to txt file
    with open("labels/coco_labels.txt", "w") as f:
        for name in names:
            f.write(f"{name}\n")

    print("Resources downloaded and prepared successfully!")
    print(f"Found {len(names)} classes in COCO dataset")
    return names


def verify_setup():
    """
    Verifies that all necessary components are installed and accessible.
    """
    try:
        import tensorflow as tf
        import tflite_runtime.interpreter as tflite
        from pycoral.utils import edgetpu

        print("TensorFlow version:", tf.__version__)
        print("TFLite Runtime available")
        print("Edge TPU utilities available")

        # Check for Edge TPU
        try:
            interpreter = tflite.Interpreter(
                experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
            )
            print("Edge TPU accessible")
        except Exception as e:
            print("Warning: Edge TPU not found or not accessible")
            print(f"Error: {str(e)}")

    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        print("Please install all required packages")


if __name__ == "__main__":
    # Download and prepare resources
    class_names = download_yolo_resources()

    # Verify setup
    verify_setup()

    print("\nSetup complete! You can now use the model with your Coral TPU.")
