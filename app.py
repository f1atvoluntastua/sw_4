# SPDX-License-Identifier: Apache-2.0
from flask import Flask, render_template, request
import os
import cv2
import onnxruntime as ort
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("..")
from box_utils import predict

app = Flask(__name__,  static_folder='.')
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-640 onnx model
face_detector_onnx = "env/version-RFB-640.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
face_detector = ort.InferenceSession(face_detector_onnx)

# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


# crop image
def cropImage(image, box):
    num = image[box[1] : box[3], box[0] : box[2]]
    return num


# face detection method
def faceDetector(orig_image, threshold=0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(
        orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold
    )
    return boxes, labels, probs

# Load the image
# image_path = 'env/chosen.jpg'  # replace with your image path
# image = cv2.imread(image_path)

# Run the face detector
#boxes, labels, probs = faceDetector(image)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            boxes, labels, probs = faceDetector(image)
            # here you would typically draw boxes on image and save the result
            # then send it to the client
            # Draw the bounding boxes on the image

            for box in boxes:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
    return render_template('index.html')



# Display the image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

if __name__ == "__main__":
    port = int(os.environ.get('WEBSITES_PORT', 8000))
    app.run(port=port)
