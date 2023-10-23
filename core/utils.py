import re

import cv2
import easyocr

reader_bn = easyocr.Reader(['bn'])
reader_en = easyocr.Reader(['en'])


def crop_image(path):
    image = cv2.imread(path)
    face_cascade = cv2.CascadeClassifier('./core/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    try:
        x, y, w, h = faces[0]
        cropped_image = image[y - int(image.shape[0] * 0.07):y + h + 2000, x + w:x + 8000]
        return cropped_image
    except:
        return "can not detect the face. Please take a more clear picture"


def preprocess_image(image):
    if image is None:
        print("Could not open or find the image")
    else:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur the image using gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        image = clahe.apply(blurred_image)
        _, image = cv2.threshold(image, thresh=130, maxval=220, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

        image = cv2.copyMakeBorder(
            src=image,
            top=20,
            bottom=20,
            left=20,
            right=20,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255))

    return image


def get_personal_info(image):
    result_bn = reader_bn.readtext(image)
    result_en = reader_en.readtext(image)

    bn_filter = ['গণপ্রজাতন্ত্রী বাংলাদেশ সরকার', 'জাতীয় পরিচয়পত্র', 'জাতীয়', 'পরিচয়', 'পত্র', 'পিতা',
                 'মাতা', 'পিতা:', 'মাতা:', 'নাম', 'নাম:']
    pattern_bn = ['গণপ্রজাতন্ত্রী বাংলাদেশ সরকার', 'বাংলাদেশ', 'জাতীয় পরিচয়পত্র', 'জাতীয়', 'পরিচ', 'পত্র']
    compiled_patterns_bn = [re.compile(pattern) for pattern in pattern_bn]

    filtered_bn = [i for i in result_bn if (i[-1] > 0.30 and i[1] not in bn_filter)]
    filtered_bn = [i for i in filtered_bn if len(list(i[1])) >= 5]
    filtered_bn = [i for i in filtered_bn if not any(pattern.search(i[1]) for pattern in compiled_patterns_bn)]

    en_filter = ['Name', 'Name:', 'NATIONAL ID CARD', 'NATIONAL', 'ID CARD', 'ID Card']
    pattern_en = ['NATIONAL', 'National', 'ID CARD', 'ID Card']
    compiled_patterns_en = [re.compile(pattern) for pattern in pattern_en]

    bn_name_pattern = re.compile('^নাম:')
    matches_bn = bn_name_pattern.search(filtered_bn[0][1])
    if matches_bn:
        bn_name = filtered_bn[0][1].lstrip('নাম:')
        bn_name = bn_name.strip(' ')
    else:
        bn_name = filtered_bn[0][1]

    filtered_en = [i for i in result_en if (i[-1] > 0.20 and i[1] not in en_filter)]
    filtered_en = [i for i in filtered_en if len(list(i[1])) > 5]
    filtered_en = [i for i in filtered_en if not any(pattern_en.search(i[1]) for pattern_en in compiled_patterns_en)]

    first_item_valid = re.compile('\d+')
    while first_item_valid.search(filtered_en[0][1]):
        filtered_en.pop(0)

    en_name_pattern = re.compile('^Name:')
    matches_en = en_name_pattern.search(filtered_en[0][1])

    if matches_en:
        en_name = filtered_en[0][1].lstrip('Name:')
        en_name = en_name.strip(' ')
    else:
        en_name = filtered_en[0][1]

    pattern_id = re.compile(r'\d+')
    nid = pattern_id.findall(filtered_en[-1][1])
    nid = ''.join(nid)

    return {'bn_name': bn_name, 'en_name': en_name, 'nid': nid}
