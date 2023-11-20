import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

from core.utils import card, rotate_image
from core.utils import crop_image, get_old_info, get_smart_info

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class FileNotUploadedResponse(Exception):
    def __init__(self, message: str = "Error!"):
        self.message = message


@app.exception_handler(FileNotUploadedResponse)
async def unicorn_exception_handler(request: Request, exc: FileNotUploadedResponse):
    return JSONResponse(
        status_code=422,
        content={"message": exc.message},
    )



@app.post("/search-nid")
async def search_personal_info(file: UploadFile = File(None, description='upload a image file with extension [.png, '
                                                                         '.jpg, .jpeg]')):
    if file is None:
        raise FileNotUploadedResponse("Please provide a nid image")

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
