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
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 10, 7, 15)
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
        # result_bn = reader_bn.readtext(image)
        result_en = OCRModel.reader_en.readtext(self.image)

        # Filter bangla ocr results
        # filtered_bn = [i for i in result_bn if (i[-1] > 0.30 and len(i[1].strip()) > 5 )]

        # Filter english ocr results
        # TODO: case-1: if any text not in image
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

        en_name = filtered_en[0][1].strip().lstrip('Name:').lstrip('Name.').lstrip('Name').strip().replace(':', '.')

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

    def bday_nid_parser_easyOCR(self, image: ndarray, field):
        result = OCRModel.reader_en.readtext(self.preprocess_image(image), detail=1)
        filtered_list = [i for i in result if i[-1] > 0.3]
        filtered_list = [i for i in filtered_list if len(list(i[1])) > 7]

        if len(filtered_list) >= 1:
            if field == 'bd':
                pattern_bday = re.compile(r'.*?(\d.*)')
                matches_bday = pattern_bday.search(filtered_list[0][1])

                if matches_bday:
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

    def get_info(self):
        y, x, _ = self.image.shape

        # Segment the image
        # bn_name_seg = image[0:int(y * 0.17), 0:int(x * 0.65)]
        en_name_seg = self.image[int(y * 0.18):int(y * 0.32), 0:int(x * 0.65)]
        # f_name_seg = image[int(y * 0.33):int(y * 0.53), 0:int(x * 0.65)]
        # m_name_seg = image[int(y * 0.51):int(y * 0.69), 0:int(x * 0.65)]
        b_day_seg = self.image[int(y * 0.65):int(y * 0.85), int(x * 0.21):int(x * 0.80)]
        nid_seg = self.image[int(y * 0.77):int(y * 0.98), int(x * 0.21):x]

        # get names
        # bn_name = name_parser(bn_name_seg, lang_type='bangla')
        en_name = self.name_parser_easyOCR(en_name_seg, lang_type=LangType.english)
        # f_name = name_parser(f_name_seg, lang_type='bangla')
        # m_name = name_parser(m_name_seg, lang_type='bangla')

        # get birthdate and NID
        bday = self.bday_nid_parser_easyOCR(b_day_seg, field='bd')
        nid = self.bday_nid_parser_easyOCR(nid_seg, field='nid')

        return NIDInfo(**{'name': en_name, 'dob': bday,
                          'nid': nid})
