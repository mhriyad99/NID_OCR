from core.utils import crop_image, preprocess_image, get_personal_info

cropped_image = crop_image('D:\\Data\\photo\\images.jpg')
if type(cropped_image) != str:
    image = preprocess_image(cropped_image)
    info = get_personal_info(image)
    print(info)
else:
    print(cropped_image)

