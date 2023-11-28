## <p style='text-align: center;'> NID-OCR </p>
### Introduction 
<p style='text-align: justify;'>In this project, we tried to develop a backend-system that performs Optical
Character Recognition (OCR) on the National Identification (NID) card to extract information from it.
Specifically, we are trying to get the english name of the person, his birthdate and NID number.</p>

### Technology Used
* **Python** (Programming Language) 
* **OpenCV:** This is a popular image processing and computer vision python library
* **YOLOv8:** A powerful object detection model published by ultralytics
* **EasyOCR:** A python library for performing OCR on images
* **FastAPI:** A modern, fast (high-performance), web framework for building APIs with Python 
### Project Folder Structure
<pre style='text-align: left; padding-left: 9rem'>  
NID-OCR
|--------- core
|          |---- __inti__.py
|          |---- events.py
|          |---- utils.py
|          |---- services.py
|
|--------- models
|          |---- __init__.py
|          |---- [your_card_detection_model].pt
|
|--------- schema
|          |---- __init__.py
|          |---- common.py
|
|--------- main.py</pre>
### Life Cycle of an Incoming Request
<pre style='text-align: center;'>
Input Image
|
NID Card Detection (Smart or Old)
|
Crop Information section
|
Pre-process cropped segment
|
Apply OCR
|
Information Response
</pre>
***Input Image:*** First the user uploads his/her NID card image and sends it with the
request to the backend server for processing. If the incoming request don't contain
any image an error message is showed to the user.

***NID Card Detection*:** Upon receiving the incoming request containing an image, the system initiates
card detection by employing the card detection model stored in the `models` folder, adhering to PyTorch conventions 
with the `.pt` extension. The card detection model, based on the YOLOv8 object detection architecture
developed by ultralytics and accessible at [ultralytics](https://github.com/ultralytics/ultralytics),
has undergone training specifically for two types of National ID (NID) images: Smart cards (labeled as `1`
for new) and Old cards (labeled as `3` for old). The primary function of this model is to identify and 
delineate the bounding box of the card in the image, subsequently classifying it as either a Smart 
card `1` or an Old card `3`. Then using the bounding box we crop the card from the image. The `card` 
function in the `utils.py` does the whole thing. It receives the image as input and returns cropped card
image and card type as output.

***Crop Information Segment:*** Based on the card type either of two `class` is invoked i.e. `OldCardProcess`
or `NewCardProcess`. After, the card image is cropped again using the `crop` method from the `class` to strip 
away all the unnecessary text from the image keeping only the person's information.

***Pre-Process the Segment:*** 
