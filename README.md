\# Face Anti-Spoofing



A real-time face liveness detection system that uses computer vision and YOLO to distinguish between real faces and spoofing attempts.



\## Overview



The project uses a webcam to detect faces and classify them as either:



\- \*\*Real\*\*

\- \*\*Fake\*\*



The model is trained using labeled real and spoofed facial samples and then used for real-time webcam-based detection.



\## Workflow



\### 1. Data Collection



`dataCollection.py` is used to collect facial images through the webcam.



Each collected sample is assigned a class:



\- `0` = Fake

\- `1` = Real



The collected images and their YOLO label files are initially stored in the data collection folder.



For better organization, real and fake samples are collected separately and then moved into their respective folders before being combined into the complete dataset.



\### 2. Dataset Preparation



`splitDATA.py` organizes the prepared dataset into training, validation, and testing sets.



The default split is:



\- 70% training

\- 20% validation

\- 10% testing



The script also generates the YOLO dataset configuration file.



\### 3. Model Training



`train.py` uses YOLOv8 to train the model on the prepared dataset.



After training, YOLO generates training results and model weights. The `best.pt` weight produced from the training run is copied into the `models` directory and can be renamed for use by the application.



\### 4. Real-Time Detection



`main.py` loads the trained model and processes webcam frames in real time.



The system displays:



\- Face bounding boxes

\- Real/Fake classification

\- Confidence score

\- Real-time processing output



\## Technologies Used



\- Python

\- OpenCV

\- CVZone

\- MediaPipe

\- Ultralytics YOLO



\## Project Structure



```text

Face-Anti-Spoofing/

├── models/

│   └── version1\_3.pt

├── Testing Scripts/

│   ├── faceDetector.py

│   ├── textFILETest.py

│   └── YOLOtest.py

├── main.py

├── dataCollection.py

├── splitDATA.py

├── train.py

├── pyproject.toml

├── uv.lock

└── README.md

