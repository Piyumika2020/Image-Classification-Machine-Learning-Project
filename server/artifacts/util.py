
import joblib
import json
import numpy as np
import base64
import cv2
import os
from wavelet import w2d
import pickle

__model = None
__class_name_to_number = {}
__class_number_to_name = {}

'''def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    # if no faces found
    if len(imgs) == 0:
        return [{
            'class': 'unknown',
            'class_probability': 0,
            'class_dictionary': __class_name_to_number
        }]

    for img in imgs:

        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scalled_raw_img.reshape(32 * 32 * 3, 1),
            scalled_img_har.reshape(32 * 32, 1)
        ))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)

        # prediction
        probs = __model.predict_proba(final)[0]
        pred_class = np.argmax(probs)
        max_prob = np.max(probs)

        # CONFIDENCE THRESHOLD
        if max_prob < 0.6:
            result.append({
                'class': 'unknown',
                'class_probability': round(max_prob * 100, 2),
                'class_dictionary': __class_name_to_number
            })
        else:
            result.append({
                'class': class_number_to_name(pred_class),
                'class_probability': [round(p * 100, 2) for p in probs],
                'class_dictionary': __class_name_to_number
            })

    return result'''


def classify_image(image_base64_data, file_path=None):
    #pass

# combining two images (raw image + haar image)
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        #if img is None:
        #    continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

#model loading
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]
    #return __class_number_to_name.get(class_num, "Unknown")

#print("Predicted class number:", class_num)
print("Available class numbers:", __class_number_to_name)



'''def load_saved_artifacts():

    print("loading saved artifacts...start")

    print("Current working directory:", os.getcwd())
    print("util.py location:", os.path.abspath(__file__))

    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./MODEL/save_class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./MODEL/save_saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")'''



def load_saved_artifacts():
    print("loading saved artifacts...start")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Base dir (util.py):", base_dir)

    class_dict_path = os.path.join(
        base_dir,
        "MODEL",
        "save_class_dictionary.json"
    )

    print("Class dictionary path:", class_dict_path)

    with open(class_dict_path, "r") as f:
        #global __class_name_to_number
        __class_name_to_number = json.load(f)
        global __class_number_to_name
        __class_number_to_name = {int(v): k.lower() for k, v in __class_name_to_number.items()}
        #global __class_number_to_name
        #__class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    model_path = os.path.join(
        base_dir,
        "MODEL",
        "model.pickle"
    )

    print("Model path:", model_path)

    with open(model_path, "rb") as f:
        global __model
        __model = pickle.load(f)

    print("loading saved artifacts...done")
    print("Class mapping:", __class_number_to_name)




#getting base64 image and decode it 
def get_cv2_image_from_base64_string(b64str):
    
    #credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
   # :param uri:
  #  :return:
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str
    #encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

#getting cropped face with 2 eyes
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    #face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )


    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

#getting ronaldo b64 image
def get_b64_test_image_for_ronaldo():
    with open("b64b.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()

    #print(classify_image(get_b64_test_image_for_ronaldo(), None))

    # print(classify_image(None, "./test_images/federer1.jpg"))
    # print(classify_image(None, "./test_images/federer2.jpg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
    # print(classify_image(None, "./test_images/virat2.jpg"))
    # print(classify_image(None, "./test_images/virat3.jpg")) # Inconsistent result could be due to https://github.com/scikit-learn/scikit-learn/issues/13211
    # print(classify_image(None, "./test_images/serena1.jpg"))
    # print(classify_image(None, "./test_images/serena2.jpg"))
    # print(classify_image(None, "./test_images/sharapova1.jpg"))