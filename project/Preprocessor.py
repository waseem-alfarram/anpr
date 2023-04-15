import cv2
import numpy as np
from scipy.ndimage import interpolation as inter


def preprocess(image, target_height= 300, remove_noise=True ,only_edges=False):
    if(type(image) is str):
        image = cv2.imread(image)
        
    output_image = _grayscale(image=image)
    output_image = _resize(image=output_image, height=target_height)
    output_image = _normalize(image=output_image)
    output_image = _binarize(image=output_image)
    output_image = _correct_skew(image=output_image)

    if(remove_noise):
        output_image = _remove_noise(image=output_image)
    if(only_edges):
        output_image = _canny(image=output_image)
    output_image = cv2.convertScaleAbs(output_image) # this line is important for useing the image in pytesseract 
    return output_image

def _grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
def _resize(image, height=None):
    h, w = image.shape[:2]
    aspect = w/h
    img = cv2.resize(image, dsize=(int(height*aspect), int(height)), interpolation=cv2.INTER_CUBIC)
    return img
    
def _normalize(image):
    image_array = np.array(image)
    normalized_image = image_array / 255.0
    mean = np.mean(normalized_image)
    normalized_image -= mean
    std = np.std(normalized_image)
    normalized_image /= std
    return normalized_image
    
def _binarize(image):
    _, im_bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    return im_bw
    
def _remove_noise(image):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    kernel = np.ones((3,3),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image
    
def _correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    
    return corrected
    

def _canny(image):
    image_uint8 = cv2.convertScaleAbs(image)
    return cv2.Canny(image_uint8, 100, 200)