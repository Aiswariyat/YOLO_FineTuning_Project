
# YOLOv5 Face Recognition Fine-Tuning

This application fine-tunes the YOLOv5 model for face recognition using a zipped file of images. After fine-tuning, the model will be able to recognize the faces provided as well as the usual objects YOLOv5 is designed for.
Link to the Docker Image: https://drive.google.com/file/d/1wypJ83gPAFJYM86q3Qpl5EMW8m4a2X10/view?usp=sharing

## Table of Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Docker Container](#running-the-docker-container)
- [Sample Commands](#sample-commands)
- [Output](#output)
- [Additional Notes](#additional-notes)
- [Cleanup](#cleanup)
- [License](#license)

## Requirements
- Docker installed on your local machine. Download Docker from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

## Setup
Before you begin, ensure you have Docker installed on your system. You can verify this by running:
```bash
docker --version
```

## Building the Docker Image
To build the Docker image, navigate to the directory containing the `Dockerfile` and `fine_tune_yolo.py` script. Then run the following command:

```bash
docker build -t yolov5-face-recognition .
```

This command will:
- Use the official Python base image.
- Install system dependencies (`git` and `curl`).
- Clone the YOLOv5 repository.
- Install the necessary Python packages.
- Download the YOLOv5 pre-trained weights.

## Running the Docker Container
After building the image, you can run the container using the following command:

```bash
docker run --rm yolov5-face-recognition --url <URL_TO_ZIPPED_FILE> --person_name <PERSON_NAME>
```

- Replace `<URL_TO_ZIPPED_FILE>` with the URL of the zipped file containing images of the person.
- Replace `<PERSON_NAME>` with the name of the person.

### Example
```bash
docker run --rm yolov5-face-recognition --url https://example.com/images.zip --person_name "John_Doe"
```

## Sample Commands
### 1. Building the Docker Image
```bash
docker build -t yolov5-face-recognition .
```

### 2. Running the Docker Container
```bash
docker run --rm yolov5-face-recognition --url https://example.com/images.zip --person_name "Ben_Affleck"
```

## Output
- The application will download the images, fine-tune the YOLOv5 model, and save the new model in the `output` directory within the container.
- Performance metrics for the face identification will be printed in the console during training and validation.
- The trained model file will be named `<person_name>_model.pth`.

## Additional Notes
- The Dockerfile automatically downloads the YOLOv5 repository and installs its requirements.
- The application uses the pre-trained `yolov5s.pt` weights to fine-tune on the provided face images.
- If you encounter any issues during the build or run process, ensure you have an active internet connection as the script involves downloading data and dependencies.

## Cleanup
To remove the Docker image after use, run:
```bash
docker rmi yolov5-face-recognition
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
