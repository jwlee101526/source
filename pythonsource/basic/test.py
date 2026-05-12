# FAST API Main.py

from fastapi import FastAPI
from fastapi.responses import UJSONResponse

app = FastAPI(default_response_class=UJSONResponse)