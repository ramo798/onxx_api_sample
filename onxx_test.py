import cv2
import numpy as np
import onnxruntime as ort

image = cv2.imread('mozi_1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28,28)).astype(np.float32)/255
input = np.reshape(gray, (1,1,28,28))

ort_session = ort.InferenceSession("mnist-12.onnx")

ort_inputs = {ort_session.get_inputs()[0].name: input}
ort_outs = ort_session.run(None, ort_inputs)
prediction = np.argmax(ort_outs[0], axis=1)[0]  # クラスラベルを取得

print(f"Predicted class: {prediction}")