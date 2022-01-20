import json
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel

from evaluate.model_step2 import TransformerModel


class Dosar(BaseModel):
    text: str


# Start API
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model

    device = "cpu"
    model = TransformerModel.load("trained_model/dosare_step2/")
    model.set_device(device)

@app.on_event("shutdown")
async def shutdown_event():
    pass


@app.get("/nerDosar/")
async def inference_dosar(dosar: Dosar):
    sentence = dosar.text

    results = model.predict(sentence)

    tokens = []
    tags =[]

    for result in results:
        if len(result['token']) > 2 and result['token'][0:2] == '##':
            tokens[-1] += result['token'][2:]

            isLastTagO = len(tags[-1]) < 2
            isCurrTag0 = len(result['tag']) < 2

            if  isLastTagO and isCurrTag0:
                pass # Ambele tag-uri O
            else:
                if (not isLastTagO) and (not isCurrTag0):
                    if tags[-1][2:] == result['tag'][2:]:
                        pass # Acelasi tag(diferit de O) la ambele 
                    else:
                        print('EROARE. Tag-uri din clase diferite.')
                        # LOGGER.error('EROARE. Tag-uri din clase diferite.')
                else:
                    print('EROARE. Un tag e 0, altul e o clasa.')
                    # LOGGER.error('EROARE. Un tag e 0, altul e o clasa.')
                    if not isCurrTag0:
                        tags[-1] = result['tag']
        else:
            tokens.append(result['token'])
            tags.append(result['tag'])

    response_json = {
        'tokens': tokens,
        'tags': tags
    }

    return JSONResponse(status_code=200, content=response_json)
    