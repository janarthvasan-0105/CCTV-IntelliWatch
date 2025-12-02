[BASE MODEL]

👣 Multi-Person Detection & Tracking using Faster R-CNN + Centroid Tracker
This project implements multi-person detection and tracking in videos using:
Faster R-CNN (ResNet-50 FPN) trained from scratch
Penn-Fudan Pedestrian Dataset
Centroid Tracking algorithm for assigning persistent IDs
The system detects multiple pedestrians in real time and assigns each person a unique ID, ensuring consistent tracking across video frames.


📌 Features

✔ Custom dataset loader for image + segmentation mask processing

✔ Faster R-CNN object detection model (from scratch)

✔ Extracts bounding boxes automatically from masks

✔ Real-time detection on video

✔ Centroid Tracker assigns unique, stable IDs

✔ Tracks multiple people simultaneously

✔ Fully implemented in PyTorch + OpenCV


📁 Dataset: Penn-Fudan Pedestrian Dataset
Each image has a corresponding segmentation mask where each pedestrian is encoded with a unique grayscale ID.

Example:
PNGImages/ → RGB images
PedMasks/ → color-coded instance masks

The code:
extracts unique object IDs
separates them into binary masks
calculates bounding boxes
prepares training labels (boxes, labels, area, mask metadata)


🧠 Model Architecture
✔ Faster R-CNN with ResNet-50 FPN backbone
✔ Loads architecture from torchvision
✔ Initializes without pre-trained weights
✔ Custom classification head with output classes:
✔ background
✔ person

Trained using:
✔ SGD optimizer
✔ Classification loss
✔ Bounding box regression loss
✔ RPN loss components


🧪 Training Pipeline
✔ Load dataset using PennFudanDatasetV2
✔ Convert segmentation masks → binary instance masks
✔ Extract bounding boxes for each person
✔ Create target dictionary for the model
✔ Train Faster R-CNN for 10 epochs
✔ Save trained model (multi_person_detector.pth)
✔ Loss is calculated internally by PyTorch’s detection engine.


🎯 Centroid Tracking

After detection, each bounding box is passed into the CentroidTracker:

How it works:

Computes the centroid of each detected bounding box

Compares new centroids with previously tracked ones

Uses a distance matrix to match closest objects

Assigns persistent IDs

Deregisters objects if they disappear for too long

This ensures consistent tracking even when:

People move

Appear or disappear

Temporary occlusion happens


🎥 Real-Time Video Inference

For each video frame:

Read frame using OpenCV

Convert to RGB + Tensor

Run Faster R-CNN detection

Filter predictions by confidence (>0.7)

Send bounding boxes to CentroidTracker

Draw:

Bounding boxes

Object ID labels

Display output in a live video window


▶️ How to Run
1. Train the Model
python train_model.py

2. Run Detection + Tracking
python run_video.py


Make sure to update:

VIDEO_SOURCE = "your_video_file.avi"

📌 File Structure
project/
│── PennFudanPed/
│── detector_model.ipynb
│── run_video.py
│── multi_person_detector.pth


📝 Key Python Dependencies

torch

torchvision

opencv-python

numpy

Pillow

Install them using:

pip install torch torchvision opencv-python pillow numpy

📊 Results

Accurate detection of multiple pedestrians

Smooth ID assignment

Robust tracking even under motion and partial occlusion

📣 Acknowledgements

Penn-Fudan Pedestrian Dataset

PyTorch torchvision model zoo

Standard centroid tracking algorithm

[ADVANCED MODEL]

YOLOv8 (yolov8n.pt): frame-level object detector. You use it to detect persons (class_id == 0). YOLO finds bounding boxes + confidences.

OSNet (torchreid): a person re-identification backbone. You crop each detected person and compute a feature embedding (vector) intended to represent that person’s appearance.

DeepSort (DeepSort): online multi-object tracker that uses motion + appearance cues; it assigns consistent track_ids over time.

Re-identification logic: you keep a dictionary of past embeddings (track_embedding_history) and use cosine distance to decide if a newly detected embedding is actually a previously seen person (to “reassign” IDs after occlusion).

Suspicious detector: for each track you keep history of center positions. If a track’s positions over the last SUSPICIOUS_TIME_FRAMES (5 seconds) vary by less than STILLNESS_THRESHOLD pixels → mark as SUSPICIOUS (assumed loitering / stillness).

Output: draws boxes + IDs, marks suspicious in red, writes to an output mp4 and shows a live window.
