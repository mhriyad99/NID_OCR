import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from core.events import load_model
from core.utils import card, rotate_image, preprocess_image, CardType
from core.utils import crop_image, get_old_info, get_smart_info
from core.services import OldCardProcess, NewCardProcess
from schema.common import ErrorResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def load_startup_models():
    load_model()


@app.exception_handler(ErrorResponse)
async def unicorn_exception_handler(request: Request, exc: ErrorResponse):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message},
    )


@app.post("/search-info")
async def search_personal_info(file: UploadFile = File(None, description='upload a image file with extension [.png, '
                                                                         '.jpg, .jpeg]')):
    if file is None:
        raise ErrorResponse(status_code=422, message="Please provide a nid image")

    image_bytes = await file.read()

    # Convert the image bytes to a NumPy array using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    card_image, card_type = card(image)
    card_image = rotate_image(card_image)

    if card_type == CardType.NEW.value:
        processed = NewCardProcess(card_image)
    else:
        processed = OldCardProcess(card_image)

    info = processed.process()
    return info


@app.post("/search-nid")
async def search_personal_info(file: UploadFile = File(None, description='upload a image file with extension [.png, '
                                                                         '.jpg, .jpeg]')):
    if file is None:
        raise ErrorResponse(status_code=422, message="Please provide a nid image")

    image_bytes = await file.read()

    # Convert the image bytes to a NumPy array using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    card_image, card_type = card(image)
    card_image = rotate_image(card_image)
    image = crop_image(card_image, card_type)

    if card_type == CardType.NEW.value:
        info = get_smart_info(image)
    else:
        info = get_old_info(image)
    return info
