import cv2
import pytesseract
import easyocr

config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789- '   #optimal configurations for pytesseract 
reader = easyocr.Reader(['en']) 


def extract_text(image, exp_out_len=7, drop_excess=True):
    """Recognizes the characters on a preprocessed license plate image.
    
    Keyword arguments:
    image -- A file path or a numpy array representation of the image to be predicted.
    model_used -- A Model type to specify wether to use easyocr or pytesseract (defult=Model.EASY_OCR)
    exp_out_len -- The expected output length, excluding space and "-" characters. (defult=7)
    drop_excess -- Drop any characters recognized beyond exp_out_len. (defult=True)

    Return: return_description
    """
    
    if(type(image) is str):
        image = cv2.imread(image)

    output = {"easyocr": "", "pytesseract" : ""}

    pytesseract_output = pytesseract.image_to_string(image, config=config).replace(" ", "").replace("-","")
    if (len(pytesseract_output) >= exp_out_len and drop_excess==True): 
        pytesseract_output = pytesseract_output[0:exp_out_len]
    output["pytesseract"] = pytesseract_output

    easyocr_output = reader.readtext(image, allowlist ='0123456789- ')  
    if(len(easyocr_output) > 0 and len(easyocr_output[0]) > 0):
        easyocr_output = easyocr_output[0][1].replace(" ", "").replace("-","")
        output["easyocr"] = easyocr_output[0:exp_out_len] if (len(easyocr_output) >= exp_out_len and drop_excess==True) else easyocr_output


    return output
