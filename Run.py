from project.Preprocessor import preprocess
from project.Predictor import extract_text
from project.Model import predict
import cv2

image_path = "license_plates/group1/001.jpg"

# matplotlib inline
from matplotlib import pyplot as plt
image = cv2.imread(image_path)
plt.imshow(image, cmap='gray')
plt.show()

results = predict(image)

def crop_plate(image, box):
    return image[box[1]:box[3], box[0]:box[2]]

for box in results:
    extra = crop_plate(image, box)
    preprocessed = preprocess(extra,  target_height= 300, remove_noise=True ,only_edges=False)
    plt.imshow(preprocessed, cmap='gray')
    plt.show()
    text = extract_text(preprocessed)
    print(text)
