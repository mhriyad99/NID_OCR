import re
from enum import Enum

import cv2
import easyocr
import numpy as np
import pytesseract
from numpy import ndarray
from scipy.ndimage import interpolation
from ultralytics import YOLO

reader_bn = easyocr.Reader(['bn'])
reader_en = easyocr.Reader(['en'])
model = YOLO("./models/last.pt")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ICardPreprocess:

    def __init__(self, image: ndarray):
        self.image = image

    def crop_image(self):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

    def process(self):
        self.crop_image()
        self.get_info()
        return self.image


class OldCardPreProcess(ICardPreprocess):
    def __init__(self, image: ndarray):
        super().__init__(image)

    def crop_image(self) -> ():
        pass

    def get_info(self) -> ():
        pass


class NewCardPreProcess(ICardPreprocess):
    def __init__(self, image: ndarray):
        super().__init__(image)

    def crop_image(self):
        pass

    def get_info(self):
        pass


def find_score(arr, angle):
    data = interpolation.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def rotate_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(gray_image, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    img = interpolation.rotate(img, best_angle, reshape=False, order=0)

    return img


class CardType(str, Enum):
    NEW = 1
    OLD = 3


def card(img):
    results = model.predict(img, stream=True)
    boxes = None
    for r in results:
        boxes = r.boxes

    card_type = int(boxes.cls)

    if card_type == CardType.NEW.value:
        card_type = 'new'
    elif card_type == 3:
        card_type = 'old'
    else:
        card_type = 'invalid'

    x_min, y_min, x_max, y_max, *_ = list(map(int, boxes.data[0]))
    return img[y_min:y_max, x_min:x_max], card_type


def crop_image_with_face(image, card_type):
    """
    This function uses face detection to crop the card image
    """

    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    try:

        if card_type == 'invalid':
            return 'invalid card'
        elif card_type == 'old':
            x, y, w, h = faces[0]
            cropped_image = image[y - int(image.shape[0] * 0.07):y + h + 2000, x + w:x + 8000]
            return cropped_image
        elif card_type == 'new':
            x, y, w, h = faces[0]
            cropped_image = image[y - int(image.shape[0] * 0.11):y + h + 2000,
                            int(x + w + image.shape[1] * 0.030):x + 8000]
            return cropped_image

    except Exception as e:
        raise Exception(f"Can't detect face. Please take a more clear picture")


def crop_image(image, card_type):
    y, x, _ = image.shape
    if card_type == 'old':
        image = image[int(y * 0.35):y, int(x * 0.25):x]
    else:
        image = image[int(y * 0.27):y, int(x * 0.3):x]
    return image


def preprocess_image(image, card_type):
    # Convert the image to grayscale
    image = cv2.fastNlMeansDenoisingColored(image, None, 5, 10, 7, 15)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image using gaussian blur
    # image = cv2.GaussianBlur(image, (5,5), 0)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    # image = clahe.apply(gray_image)
    _, image = cv2.threshold(image, thresh=100, maxval=130, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    # if card_type == 'old':
    #     # image = cv2.GaussianBlur(image, (5,5), 0)
    #     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    #     # image = clahe.apply(image)
    #     image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,2)

    # sharpening_kernel = np.array([[-1, -1, -1],
    #                           [-1,  9, -1],
    #                           [-1, -1, -1]])

    # image = cv2.filter2D(image , -1, sharpening_kernel)

    # image = cv2.copyMakeBorder(
    # src=image,
    # top=20,
    # bottom=20,
    # left=20,
    # right=20,
    # borderType=cv2.BORDER_CONSTANT,
    # value=(255, 255, 255))

    return image


def name_parser_tesseract(image, lang_type, card_type='new'):
    myconfig = r"--psm 6 --oem 3"

    if lang_type == 'bangla':
        infopy = pytesseract.image_to_string(preprocess_image(image, card_type), lang='ben', config=myconfig)
        pattern = r'[A-Za-z0-9\u09E6-\u09EF]'
    else:
        infopy = pytesseract.image_to_string(preprocess_image(image, card_type), lang='eng', config=myconfig)
        pattern = r'[0-9\u09E6-\u09EF]'

    info_list = infopy.split('\n')
    filtered_info = [i for i in info_list if len(i) > 5]
    sub_pattern = '[!@#\$%^&*৷()_+{}[\]:;<>,?\/\\=|`~"\'-]'

    # Filter out strings that match the pattern
    filtered_strings = [s for s in filtered_info if not re.search(pattern, s)]
    filtered_strings = [re.sub(sub_pattern, '', s) for s in filtered_strings]

    if len(filtered_strings) >= 1:
        return filtered_strings[0]
    else:
        return ''


def name_parser_easyOCR(image, lang_type, card_type='new'):
    if lang_type == 'bangla':
        result = reader_bn.readtext(preprocess_image(image, card_type), detail=1)
        pattern = r'[A-Za-z0-9\u09E6-\u09EF]'
    else:
        result = reader_en.readtext(preprocess_image(image, card_type), detail=1)
        pattern = r'[0-9\u09E6-\u09EF]'

    sub_pattern = '[!@#\$%^&*()_+{}[\]:;<>,?\/\\=|`~"\'-]'
    filtered_list = [i for i in result if i[-1] > 0.3]
    filtered_list = [i for i in filtered_list if len(list(i[1])) > 5]
    filtered_strings = [s[1] for s in filtered_list if not re.search(pattern, s[1])]
    filtered_strings = [re.sub(sub_pattern, '', s) for s in filtered_strings]

    if len(filtered_strings) >= 1:
        return filtered_strings[0]
    else:
        return ''


def bday_nid_parser_easyOCR(result, field):
    filtered_list = [i for i in result if i[-1] > 0.3]
    filtered_list = [i for i in filtered_list if len(list(i[1])) > 5]

    if len(filtered_list) >= 1:
        if field == 'bd':
            pattern_bday = re.compile(r'.*?(\d.*)')
            matches_bday = pattern_bday.search(filtered_list[0][1])

            if matches_bday:
                bday = filtered_list[0][1]
                bday = matches_bday.group(1)
                bday = bday.strip(' ')
            else:
                bday = ''

            return bday

        elif field == 'nid':
            pattern_id = re.compile(r'\d+')
            nid = pattern_id.findall(filtered_list[-1][1])
            nid = ''.join(nid)

            return nid


def get_smart_info(image, name_parser=name_parser_easyOCR, bday_nid_parser=bday_nid_parser_easyOCR):
    y, x = image.shape[0:2]

    # Segment the image
    # bn_name_seg = image[0:int(y * 0.17), 0:int(x * 0.65)]
    en_name_seg = image[int(y * 0.18):int(y * 0.32), 0:int(x * 0.65)]
    # f_name_seg = image[int(y * 0.33):int(y * 0.53), 0:int(x * 0.65)]
    # m_name_seg = image[int(y * 0.51):int(y * 0.69), 0:int(x * 0.65)]
    b_day_seg = image[int(y * 0.65):int(y * 0.85), int(x * 0.21):int(x * 0.80)]
    nid_seg = image[int(y * 0.77):int(y * 0.98), int(x * 0.21):x]

    # cv2.imwrite('bn_name_seg.jpg', bn_name_seg)
    # cv2.imwrite('en_name_seg.jpg', en_name_seg)
    # cv2.imwrite('f_name_seg.jpg', f_name_seg)
    # cv2.imwrite('m_name_seg.jpg', m_name_seg)
    # cv2.imwrite('b_day_seg.jpg', b_day_seg)
    # cv2.imwrite('nid_seg.jpg', nid_seg)

    # get names
    # bn_name = name_parser(bn_name_seg, lang_type='bangla')
    en_name = name_parser(en_name_seg, lang_type='english')
    # f_name = name_parser(f_name_seg, lang_type='bangla')
    # m_name = name_parser(m_name_seg, lang_type='bangla')

    # get birthdate
    result_bd = reader_en.readtext(preprocess_image(b_day_seg, card_type='new'), detail=1)
    bday = bday_nid_parser(result_bd, field='bd')

    # get NID
    result_nid = reader_en.readtext(preprocess_image(nid_seg, card_type='new'), detail=1)
    nid = bday_nid_parser(result_nid, field='nid')

    return {'en_name': en_name, 'Date of birth': bday, 'nid': nid}


def get_old_info(image):
    # result_bn = reader_bn.readtext(image)
    result_en = reader_en.readtext(image)

    # Filter bangla ocr results
    # filtered_bn = [i for i in result_bn if (i[-1] > 0.30 and len(i[1].strip()) > 5)]

    # Filter english ocr results
    filtered_en = [i for i in result_en if (i[-1] > 0.35 and len(i[1].strip()) > 5)]

    # Individual's bangla name
    # bn_name = filtered_bn[0][1].strip().lstrip('নাম:').lstrip('নাম.').lstrip('নাম').strip()

    # Father's name
    # f_name = filtered_bn[1][1].strip().lstrip('পিতা:').lstrip('পিতা.').lstrip('পিতা').strip()

    # Mother's name
    # m_name = filtered_bn[2][1].strip().lstrip('মাতা:').lstrip('মাতা.').lstrip('মাতা').strip()

    # Individual's english name
    first_item_valid = re.compile('\d+')

    while first_item_valid.search(filtered_en[0][1]):
        filtered_en.pop(0)

    en_name = filtered_en[0][1].strip().lstrip('Name:').lstrip('Name.').lstrip('Name').strip()
    en_name = en_name.replace(':', '.')

    filtered_en = [i for i in filtered_en if len(list(i[1])) > 8]

    # Date of birth
    pattern_bday = re.compile(r'.*?(\d.*)')
    matches_bday = pattern_bday.search(filtered_en[-2][1])

    if matches_bday:
        bday = matches_bday.group(1)
        bday = bday.strip(' ')
    else:
        bday = ''

    # Individual's NID no
    pattern_id = re.compile(r'\d+')
    nid = pattern_id.findall(filtered_en[-1][1])
    nid = ''.join(nid)

    return {'en_name': en_name,'Date of birth': bday, 'nid': nid}


def get_personal_info_pytesseract(image):
    # bangla info extract
    myconfig = r"--psm 6 --oem 3"
    info_ben = pytesseract.image_to_string(image, lang='ben', config=myconfig)

    info_ben_list = info_ben.split('\n')
    filtered_info_ben = [i for i in info_ben_list if len(i) > 5]

    pattern_ben = r'[A-Za-z0-9\u09E6-\u09EF]'
    sub_pattern_ben = '[!@#\$%^&*()৷_+{}[\]:;<>,?\/\\=|`~"\'-]'
    filtered_strings_ben = [s for s in filtered_info_ben if not re.search(pattern_ben, s)]
    filtered_strings_ben = [re.sub(sub_pattern_ben, '', s) for s in filtered_strings_ben]

    # Individual's Name
    if len(filtered_strings_ben) >= 1:
        bn_name = filtered_strings_ben[0].lstrip('নাম:').lstrip('নাম.').lstrip('নাম').strip()
    else:
        bn_name = ''

    # Father's Name
    if len(filtered_strings_ben) >= 2:
        f_name = filtered_strings_ben[1].lstrip('পিতা:').lstrip('পিতা.').lstrip('পিতা').strip()
    else:
        f_name = ''

    # Mother's Name
    if len(filtered_strings_ben) >= 3:
        m_name = filtered_strings_ben[2].lstrip('মাতা:').lstrip('মাতা.').lstrip('মাতা').strip()
    else:
        m_name = ''

    # english info extract
    y, x = image.shape[0:2]
    image = image[int(y * 0.17):y, 0:x]
    result_en = reader_en.readtext(image, detail=1)

    en_filter = ['Name', 'Name:', 'NATIONAL ID CARD', 'NATIONAL', 'ID CARD', 'ID Card']
    pattern_en = ['NATIONAL', 'National', 'ID CARD', 'ID Card']
    compiled_patterns_en = [re.compile(pattern) for pattern in pattern_en]

    # Filter english ocr results
    filtered_en = [i for i in result_en if (i[-1] > 0.35 and i[1] not in en_filter)]
    filtered_en = [i for i in filtered_en if len(list(i[1])) > 5]
    filtered_en = [i for i in filtered_en if not any(pattern_en.search(i[1]) for pattern_en in compiled_patterns_en)]

    # Individual's english name
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
    en_name = en_name.lstrip('Name.').strip().replace(':', '.')

    # Date of birth
    pattern_bday = re.compile(r'.*?(\d.*)')
    matches_bday = pattern_bday.search(filtered_en[-3][1])

    if matches_bday:
        bday = filtered_en[-3][1]
        bday = matches_bday.group(1)
        bday = bday.strip(' ')
    else:
        bday = ''

    # Individual's NID no
    pattern_id = re.compile(r'\d+')
    nid = pattern_id.findall(filtered_en[-1][1])
    nid = ''.join(nid)

    return {'bn_name': bn_name, 'en_name': en_name, 'father': f_name, 'mother': m_name, 'Date of birth': bday,
            'nid': nid}
