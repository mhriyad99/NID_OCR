from enum import Enum

from pydantic import BaseModel


class ErrorResponse(Exception):
    def __init__(self, status_code: int = 404, message: str = "Error!"):
        self.message = message
        self.status_code = status_code


class NIDInfo(BaseModel):
    name: str
    dob: str
    nid: str


class CardType(int, Enum):
    NEW = 1
    OLD = 3


class LangType(str, Enum):
    bangla = 'ben'
    english = 'eng'
