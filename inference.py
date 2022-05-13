from html import entities
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
    offset_mapping = []

    # Tokens + Tags
    for result in results:
        if len(result['token']) > 2 and result['token'][0:2] == '##':
            tokens[-1] += result['token'][2:]
            offset_mapping[-1][1] = result['offset_mapping'][1]

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
            offset_mapping.append([result['offset_mapping'][0],result['offset_mapping'][1]])

    # Entities
    entities = {}
    last_entity = ''
    last_idx = 0

    for i in range(len(tags)):
        if len(tags[i]) > 1:
            idx_start, idx_end = offset_mapping[i]

            if tags[i][0] == 'B' or last_entity != tags[i][2:]:
                if tags[i][2:] not in entities:
                    entities[tags[i][2:]] = []
                entities[tags[i][2:]].append(sentence[idx_start:idx_end])
            else:
                if tags[i][2:] not in entities:
                    # Nu ar trb sa intre pe aici tho
                    # LOGGER.error('EROARE. Nu ar trebui sa intre in aceasta parte de cod.')
                    entities[tags[i][2:]] = ['']
                entities[tags[i][2:]][-1] = entities[tags[i][2:]][-1] + sentence[last_idx:idx_start] + sentence[idx_start:idx_end]

            last_entity = tags[i][2:]
            last_idx = idx_end

    # Create the response body
    response_json = {
        'tokens': tokens,
        'tags': tags,
        'entities': entities
    }

    return JSONResponse(status_code=200, content=response_json)
    