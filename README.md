# NID-OCR
## Installation
### Local setup
**Step 1**: Install Python if not already installed 

**Step 2**: Install python virtualenv package
```shell
pip install virtualenv
```
**Step 3**: Open Command Prompt and navigate to the desired directory
```shell
cd path\to\your\project
```
**Step 4**: Create a new virtual environment
```shell
python -m venv env
```
**Step 5**: Activate the virtual environment  
```shell
venv\Scripts\activate
```
**Step 6**: Install dependencies using pip within the virtual environment
```shell
pip install -r requirements.txt
```
**Step 7**: Run server
```shell
uvicorn main:app
```
