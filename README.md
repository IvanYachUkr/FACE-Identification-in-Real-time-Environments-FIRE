# Face Recognition System Implementation

## Overview

This project focuses on the development of a Python-based, real-time face recognition system designed for dynamic environments, particularly in robotics. The pipeline includes:

- **Face Detection:** Utilizes MediaPipe, YuNet, and RetinaFace to balance speed and accuracy.
- **Face Encoding:** Employs FaceNet128 for speed and FaceNet512 for precision.
- **Face Tracking:** Implements a custom SORT-N algorithm for robust tracking, including non-linear dynamics.
- **Database Management:** Uses SQLite with optional AES-256 encryption for secure storage.
- **Similarity Search:** Integrates HNSWlib for fast and scalable in-memory embedding comparison.

## Pipeline Components

### 1. Face Detection

The system incorporates three face detectors:

- **MediaPipe:** Optimized for high-speed detection with acceptable accuracy. Ideal for time-critical applications.
- **YuNet:** Balances accuracy and speed, suitable for applications requiring moderate computational resources.
- **RetinaFace:** Provides the highest accuracy but is computationally intensive, making it more suitable for offline processes.

### 2. Face Encoding

Two variants of FaceNet are used:

- **FaceNet128:** Offers rapid encoding with lower dimensional embeddings, suitable for real-time applications.
- **FaceNet512:** Provides higher precision, ideal for scenarios where accuracy is prioritized over speed.

### 3. Face Tracking

To reduce computational load and maintain identity across frames:

- **SORT-N Algorithm:** Enhances the original SORT algorithm by incorporating:
  - Non-linear motion handling using an Unscented Kalman Filter (UKF).
  - Logarithmic transformations for stable scale and aspect ratio estimation.
  - A combined cost metric using IoU and Euclidean distance for better data association.
  - Expanded state vector with acceleration components for improved handling of abrupt movements.

### 4. Similarity Search

- **HNSWlib:** Enables efficient approximate nearest neighbor search for embeddings, supporting dynamic updates without re-indexing.

### 5. Database Management

- **SQLite:** Lightweight and portable for embedding storage and management.
- **AES-256 Encryption:** Ensures secure handling of sensitive data with salted passwords.

## Performance Metrics

### Detection and Encoding Speed

- **MediaPipe + FaceNet128:** Achieves up to 87.81 FPS in real-time applications.
- **YuNet + FaceNet512:** Balances speed (30+ FPS with frame skipping) and accuracy.
- **RetinaFace:** Suitable for offline use with high AUC of 0.991 but low FPS.

### Accuracy

- Cosine similarity was chosen as the similarity metric, consistently yielding higher AUC scores compared to Euclidean distance.
- Pipelines are evaluated on datasets like CASIA-FaceV5 and open-source videos, demonstrating strong performance in both speed and recognition accuracy.

## Requirements

The system requires the following dependencies:

```
cryptography==44.0.0
filterpy==1.4.5
hnswlib==0.8.0
mediapipe==0.10.20
numpy==1.26.4
onnxruntime==1.20.1
psutil==6.1.1
scipy==1.14.1
```

Ensure all dependencies are installed for optimal performance.

## Usage Guide

### Installation
1. Clone the repository and navigate to the project directory:
   ```
   git clone https://github.com/IvanYachUkr/-FACE-Identification-in-Real-time-Environments-FIRE-.git
   cd <repository_directory>
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the System

#### Image Mode
To process a single image:
```bash
python main.py --mode image --input /path/to/image.jpg --save /path/to/output.jpg --detector mediapipe --encoder 128 --password yourpassword --log
```

#### Video Mode
To process a video file:
```bash
python main.py --mode video --input /path/to/video.mp4 --save /path/to/output.avi --detector yunet --encoder 512 --log
```

#### Webcam Mode
To process live webcam feed:
```bash
python main.py --mode webcam --save /path/to/output.avi --show --annotate --password yourpassword --label "Person_Name"
```

### Additional Options

| **Option**                   | **Description**                                                                                                  |
|------------------------------|------------------------------------------------------------------------------------------------------------------|
| `--mode`                     | Specifies the mode of operation: `image`, `video`, or `webcam`.                                                 |
| `--input`                    | Path to the input image or video file (not required for webcam mode).                                           |
| `--save`                     | Path to save the annotated output (image or video).                                                             |
| `--label`                    | Label or name to add to new faces (image mode) or rename recognized faces (webcam mode).                       |
| `--log`                      | Enables detailed logging.                                                                                       |
| `--show`                     | Displays processed frames in a window during execution.                                                         |
| `--password`                 | Password for encrypting/decrypting database files and outputs.                                                  |
| `--detector`                 | Specifies the face detection model: `mediapipe`, `yunet`, or `retinaface`.                                      |
| `--encoder`                  | Specifies the encoder type: `128` (Facenet128) or `512` (Facenet512).                                           |
| `--detection_interval`       | Specifies the number of frames to skip for face detection (use `1` for processing all frames).                   |
| `--core`                     | Restricts the program to a single core if set to `1`.                                                            |
| `--annotate`/`--no-annotate` | Toggles the drawing of bounding boxes and labels on images/videos.                                              |
| `--hnsw_index_path`          | Custom path for the HNSWlib index file.                                                                         |
| `--hnsw_labels_path`         | Custom path for the HNSWlib labels file.                                                                        |
| `--hnsw_db_ids_path`         | Custom path for the HNSWlib database IDs file.                                                                  |
| `--sqlite_db_path`           | Custom path for the SQLite database file (unencrypted).                                                         |
| `--sqlite_db_encrypted_path` | Custom path for the encrypted SQLite database file.                                                             |
| `--interested_label`         | If set, filters the recognition results to only track and recognize faces with the specified label.              |

### Managing Databases
- **SQLite Database Path:** Specify `--sqlite_db_path` to set a custom SQLite database file path.
- **Encryption:** Use `--password` for encrypting/decrypting database files and outputs.

For additional help, use:
```bash
python main.py --help
```

