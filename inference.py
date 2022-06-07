from html import entities
import json
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel

from model_step2 import TransformerModel


class Dosar(BaseModel):
    text: str


def clean_text(text):
    text = text.lower()

    text = text.replace('Ţ', 'Ț')
    text = text.replace('ţ', 'ț')
    text = text.replace('–', '-')
    text = text.replace('Ş', 'Ș')
    text = text.replace('ş', 'ș')
    text = text.replace('ˮ', '“')
    text = text.replace('ʼ', '’')

    return text


# Start API
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model

    device = "cpu"
    model = TransformerModel.load("trained_models/trained_model4/")
    # model = TransformerModel.load("trained_models/trained_model_test/")
    model.set_device(device)

@app.on_event("shutdown")
async def shutdown_event():
    pass


@app.get("/nerDosar/")
async def inference_dosar(dosar: Dosar):
    sentence = dosar.text
    cleaned_sentence = clean_text(sentence)

    results = model.predict(cleaned_sentence)

    tokens = []
    tags =[]
    offset_mapping = []
    # print(results)

    # Tokens + Tags
    for result in results[1:-1]:
        # print(result)
        if len(result['token']) > 2 and result['token'][0:2] == '##':
            tokens[-1] += result['token'][2:]
            offset_mapping[-1][1] = result['offset_mapping'][1]

            lastTag = tags[-1] if len(tags) else 'O'
            currTag = result['tag']

            # Same class
            if lastTag[2:] == currTag[2:]:
                pass
            else:
                if lastTag == 'O' and currTag[0] == 'I':
                    pass
                elif lastTag[0] == 'I' and currTag[0] == 'O':
                    pass
                else:
                    print('Token continuu, combinatie ciudata')
        else:
            lastTag = tags[-1] if len(tags) else 'O'
            currTag = result['tag']

            tokens.append(result['token'])
            offset_mapping.append([result['offset_mapping'][0],result['offset_mapping'][1]])
            
            # Same label/class or 
            # New class or 
            # currTag O
            if lastTag[2:] == currTag[2:] or \
               currTag[0] == 'B' or \
               currTag[0] == 'O':
                tags.append(result['tag'])
            # currTag is I but different than last tag non-O
            elif lastTag[0] == 'B' and currTag[0] == 'I' or \
                 lastTag[0] == 'I' and currTag[0] == 'I':
                tags.append('I'+lastTag[1:])
            # last is O but current is I
            elif lastTag == 'O' and currTag[0] == 'I':
                tags.append(lastTag)
            else:
                print('Token individual, caz ciudat.')

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
    