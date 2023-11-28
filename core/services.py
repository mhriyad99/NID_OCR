import re

import cv2
from numpy import ndarray
from pytesseract import pytesseract

from core.utils import OCRModel
from schema.common import NIDInfo, LangType


class ICardProcess:

    def __init__(self, image: ndarray):
        self.image = image

    def crop_image(self):
        raise NotImplementedError()

    def get_info(self) -> NIDInfo:
        raise NotImplementedError()

    @staticmethod
    def preprocess_image(image):
        # De-noising the picture
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 10, 7, 15)

        # converting to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, thresh=100, maxval=130, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        return image

    def process(self):
        self.crop_image()
        info = self.get_info()
        return info


class OldCardProcess(ICardProcess):
    def __init__(self, image: ndarray):
        super().__init__(image)

    def crop_image(self):
        y, x, _ = self.image.shape
        self.image = self.image[int(y * 0.35):y, int(x * 0.23):x]

    def get_info(self):
        b_day = ''

        result_en = OCRModel.reader_en.readtext(self.preprocess_image(self.image))

        if len(result_en) < 2:
            return NIDInfo(**{'name': '', 'dob': '', 'nid': ''})

        filtered_en = [i for i in result_en if (i[-1] > 0.35 and len(i[1].strip()) > 5)]

        # Individual's english name
        first_item_valid = re.compile('\d+')

        while first_item_valid.search(filtered_en[0][1]):
            filtered_en.pop(0)

        en_name = filtered_en[0][1].strip()
        words_to_remove = ['Name', 'lame', 'ame', ':', '.']
        for word in words_to_remove:
            if en_name.startswith(word):
                en_name = en_name.lstrip(word).strip()

        filtered_en = [i for i in filtered_en if len(list(i[1])) > 8]

        # Date of birth
        pattern_bday = re.compile(r'.*?(\d.*)')
        matches_bday = pattern_bday.search(filtered_en[-2][1])

        if matches_bday:
            b_day = matches_bday.group(1)
            b_day = b_day.strip(' ')

        # Individual's NID no
        pattern_id = re.compile(r'\d+')
        nid = pattern_id.findall(filtered_en[-1][1])
        nid = ''.join(nid)

        return NIDInfo(**{'name': en_name, 'dob': b_day,
                          'nid': nid})


class NewCardProcess(ICardProcess):
    def __init__(self, image: ndarray):
        super().__init__(image)

    def crop_image(self):
        y, x, _ = self.image.shape
        self.image = self.image[int(y * 0.27):y, int(x * 0.3):x]

    @staticmethod
    def get_pattern(lang_type: LangType = LangType.english):
        if lang_type.value == LangType.bangla.value:
            return r'[A-Za-z0-9\u09E6-\u09EF]'
        return r'[0-9\u09E6-\u09EF]'

    @staticmethod
    def get_sub_pattern():
        return '[!@#\$%^&*()_+{}[\]:;<>,?\/\\=|`~"\'-]'

    @staticmethod
    def get_easy_ocr_reader(lang_type: LangType = LangType.english):
        if lang_type.value == LangType.bangla.value:
            return OCRModel.reader_bn
        return OCRModel.reader_en

    def name_parser_tesseract(self, image: ndarray, lang_type: LangType = LangType.english) -> str:
        """
        image -> str
        :param image: ndarray
        :param lang_type:
        :return: str
        """
        myconfig = r"--psm 6 --oem 3"

        infopy = pytesseract.image_to_string(self.preprocess_image(image), lang=lang_type.value, config=myconfig)
        pattern = self.get_pattern(lang_type)

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

    def name_parser_easyOCR(self, image: ndarray, lang_type: LangType = LangType.english) -> str:
        """
        image -> str
        :param image:
        :param lang_type:
        :return: str
        """
        result = self.get_easy_ocr_reader(lang_type).readtext(self.preprocess_image(image), detail=1)
        if len(result) < 1:
            return ''

        pattern = self.get_pattern(lang_type)
        sub_pattern = '[!@#\$%^&*()_+{}[\]:;<>,?\/\\=|`~"\'-]'
        filtered_list = [i for i in result if i[-1] > 0.3]
        filtered_list = [i for i in filtered_list if len(list(i[1])) > 5]
        filtered_strings = [s[1] for s in filtered_list if not re.search(pattern, s[1])]
        filtered_strings = [re.sub(sub_pattern, '', s) for s in filtered_strings]

        if len(filtered_strings) >= 1:
            return filtered_strings[0]
        else:
            return ''

    def bday_parser_easyOCR(self, image: ndarray) -> str:
        result = OCRModel.reader_en.readtext(self.preprocess_image(image), detail=1)
        if len(result) < 1:
            return ''
        filtered_list = [i for i in result if i[-1] > 0.3 and len(list(i[1])) > 7]
        # filtered_list = [i for i in filtered_list if len(list(i[1])) > 7]
        bday = ''

        if len(filtered_list) >= 1:
            pattern_bday = re.compile(r'.*?(\d.*)')
            matches_bday = pattern_bday.search(filtered_list[0][1])
            if matches_bday:
                bday = matches_bday.group(1)
                bday = bday.strip(' ')

        return bday

    def nid_parser_easyOCR(self, image: ndarray) -> str:
        result = OCRModel.reader_en.readtext(self.preprocess_image(image), detail=1)
        if len(result) < 1:
            return ''
        filtered_list = [i for i in result if i[-1] > 0.3 and len(list(i[1])) > 9]
        # filtered_list = [i for i in filtered_list if len(list(i[1])) > 9]
        nid = ''

        if len(filtered_list) >= 1:
            pattern_id = re.compile(r'\d+')
            nid = pattern_id.findall(filtered_list[-1][1])
            nid = ''.join(nid)

        return nid

    def get_info(self):
        y, x, _ = self.image.shape

        # Segment the image
        en_name_seg = self.image[int(y * 0.18):int(y * 0.35), 0:int(x * 0.65)]
        b_day_seg = self.image[int(y * 0.65):int(y * 0.85), int(x * 0.21):int(x * 0.80)]
        nid_seg = self.image[int(y * 0.77):int(y * 0.98), int(x * 0.21):x]

        # get name
        en_name = self.name_parser_easyOCR(en_name_seg, lang_type=LangType.english)

        # get birthdate and NID
        bday = self.bday_parser_easyOCR(b_day_seg)
        nid = self.nid_parser_easyOCR(nid_seg)

        return NIDInfo(**{'name': en_name, 'dob': bday,
                          'nid': nid})
