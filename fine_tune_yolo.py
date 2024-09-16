import os
import sys
import zipfile
import requests
import shutil
import torch
from torchvision import transforms
from PIL import Image
import random
import argparse

sys.path.insert(0, './yolov5')
from yolov5 import train

def download_and_extract_zip(url, extract_to):
    response = requests.get(url)
    zip_path = os.path.join(extract_to, 'lfw.zip')
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)

def prepare_yolo_format(images_dir):
    # Create train and val directories
    train_dir = os.path.join(images_dir, 'train')
    val_dir = os.path.join(images_dir, 'val')
    labels_dir_train = os.path.join(train_dir, 'labels')
    labels_dir_val = os.path.join(val_dir, 'labels')

    os.makedirs(labels_dir_train, exist_ok=True)
    os.makedirs(labels_dir_val, exist_ok=True)

    # Get list of image files and split into train and val (80% train, 20% val)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Move images and create label files for the training set
    for image_file in train_files:
        src_img_path = os.path.join(images_dir, image_file)
        dest_img_path = os.path.join(train_dir, image_file)
        shutil.move(src_img_path, dest_img_path)

        # Create label file in train/labels
        label_file_name = image_file.replace('.jpg', '.txt')
        label_file_path = os.path.join(labels_dir_train, label_file_name)
        with open(label_file_path, 'w') as f:
            f.write(f"0 0.5 0.5 1.0 1.0\n")

    # Move images and create label files for the validation set
    for image_file in val_files:
        src_img_path = os.path.join(images_dir, image_file)
        dest_img_path = os.path.join(val_dir, image_file)
        shutil.move(src_img_path, dest_img_path)

        # Create label file in val/labels
        label_file_name = image_file.replace('.jpg', '.txt')
        label_file_path = os.path.join(labels_dir_val, label_file_name)
        with open(label_file_path, 'w') as f:
            f.write(f"0 0.5 0.5 1.0 1.0\n")

    return os.path.abspath(train_dir), os.path.abspath(val_dir)

def fine_tune_yolo(images_dir, person_name, output_dir, epochs=10):
    train_dir, val_dir = prepare_yolo_format(images_dir)

    data_config_path = os.path.join(output_dir, 'data.yaml')
    with open(data_config_path, 'w') as f:
        f.write(f'train: {train_dir}\n')
        f.write(f'val: {val_dir}\n')
        f.write(f'nc: 1\n')
        f.write(f"names: ['{person_name}']\n")

    train.run(
        data=data_config_path,
        imgsz=640,
        batch_size=8,
        epochs=epochs,
        weights='yolov5s.pt',
        project=output_dir,
        name=f"{person_name}_model",
        hyp='yolov5/data/hyps/hyp.scratch-low.yaml'
    )

def main(url, person_name):
    base_dir = '.'
    images_dir = os.path.join(base_dir, 'images')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    download_and_extract_zip(url, images_dir)

    person_images_dir = os.path.join(images_dir, person_name)
    if not os.path.exists(person_images_dir):
        print(f"No images found for person: {person_name}")
        return

    print(f"Fine-tuning the model on images of {person_name}...")
    fine_tune_yolo(person_images_dir, person_name, output_dir, epochs=10)
    print(f"Model training complete. Results saved in {output_dir}")

    # Optionally: Output performance metrics (bonus points)
    os.system(f'python yolov5/val.py --weights {output_dir}/{person_name}_model/weights/best.pt --data {output_dir}/data.yaml --img 640')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv5 for face recognition.")
    parser.add_argument('--url', type=str, required=True, help="URL of the zipped file containing images")
    parser.add_argument('--person_name', type=str, required=True, help="Name of the person")
    args = parser.parse_args()
    
    main(args.url, args.person_name)
