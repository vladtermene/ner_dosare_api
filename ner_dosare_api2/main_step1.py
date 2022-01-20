import json
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel

from evaluate.model_step1 import TransformerModel


class Dosar(BaseModel):
    text: str


# Start API
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model

    device = "cpu"
    model = TransformerModel.load("trained_model/dosare_first_model/")
    model.set_device(device)

@app.on_event("shutdown")
async def shutdown_event():
    pass


# @app.get("/ner_bpi_eg/")
# async def inference_bpi_eg(part: str, id_wanted: int):
#     BPI_MAPPER = {
#         'madalina': 1,
#         'george': 2,
#         'liviu': 3,
#         'andrada': 4,
#         'sergiu': 5,
#         'ioana':6
#     }
    
#     with open(f'bpis/part{BPI_MAPPER[part]}.json') as part_json_file:
#         part_json = json.load(part_json_file)

#         for bpi_doc in part_json:
#             if bpi_doc['id'] == id_wanted:
#                 input_json = bpi_doc
#                 break

#         paragraf = input_json['data']['html']

#         output_arrays = get_result(paragraf, model)

#         if output_arrays is None:
#             output_arrays = get_result_anomaly(paragraf, model)

#             if output_arrays is None:
#                 return JSONResponse(status_code=400, content={"message": "BPI could not be inferred."})
    
#     output_tokens, output_tags, entities = output_arrays

#     response_json = {
#         'tokens': output_tokens,
#         'tags': output_tags,
#         'entities': entities
#     }

#     return JSONResponse(status_code=200, content=response_json)

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
    