# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies, including git and curl
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy the Python script and requirements file into the container
COPY fine_tune_yolo.py . 
COPY requirements.txt .

# Install general dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 repository and install its dependencies
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install --no-cache-dir -r yolov5/requirements.txt

# Download YOLOv5 pre-trained weights
RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o yolov5/yolov5s.pt

# Command to run the application (default values can be overridden)
CMD ["python", "fine_tune_yolo.py", "--url", "https://drive.google.com/uc?id=1_0I5ytKxC0oGNL3VLqK7ZSwN_FtH9xmK&export=download", "--person_name", "Ben_Affleck"]
