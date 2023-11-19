# NID_OCR
## Installation
### Local setup
**Step 1**: Install Python if not already installed 

**Step 2**: Open Command Prompt and navigate to the desired directory
```shell
cd path\to\your\project
```
**Step 3**: Create a new virtual environment
```shell
python -m venv venv
```
**Step 4**: Activate the virtual environment  
```shell
venv\Scripts\activate
```
**Step 5**: Install dependencies using pip within the virtual environment
```shell
pip install requirements.txt
```
**Step 6**: Run server
```shell
uvicorn main:app
```
