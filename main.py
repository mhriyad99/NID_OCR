import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse

from core.utils import card, rotate_image
from core.utils import crop_image, get_old_info, get_smart_info

app = FastAPI()


@app.post("/")
async def search_personal_info(file: UploadFile = File(..., description='upload a image file with extension [.png, '
                                                                        '.jpg, .jpeg]')):
    # Read the uploaded image as bytes
    image_bytes = await file.read()

    # Convert the image bytes to a NumPy array using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    card_image, card_type = card(image)
    card_image = rotate_image(card_image)
    image = crop_image(card_image, card_type)

    if card_type == 'new':
        info = get_smart_info(image)
    else:
        info = get_old_info(image)
    return info





