import json
import stanza
import string
from progress.bar import IncrementalBar
import sys


nlp = stanza.Pipeline(lang='ro', processors='tokenize', tokenize_no_ssplit=True)

"""
['annotations'][array]['result][array]['value']['start']
['annotations'][array]['result][array]['value']['end']
['annotations'][array]['result][array]['value']['text']
['annotations'][array]['result][array]['value']['labels']['array']
[
  {
    "id": 1,
    "text": "El",
    "start_char": 0,
    "end_char": 2
  }
]
"""

LABEL_MAPPER = {
    'PF': 1,
    'PF_reprezentat': 1,
    'PF_delegat': 1,
    'PJ': 3,
    'PJ_reprezentat': 3,
    'PJ_delegat': 3,
    'STAT': 5,
    'STAT_reprezentat': 5,
    'STAT_delegat': 5,
    'Locatie_PJ': 7
}
ROM_SPECIAL_CHARS = 'ǎǍăĂâÂȃȂșȘîÎțȚãÃȋȊ' 
PUNCTUATION_SPECIAL_CHARS = '„”“«»’‘…´″‚'
MONEY_CHARS = '€$£'

ALLOWED_CHARS = string.printable + ROM_SPECIAL_CHARS + PUNCTUATION_SPECIAL_CHARS + MONEY_CHARS


# Concatenate input jsons
json_dataset = []
for nume_fisier in ['bogdan', 'ioana', 'madalina', 'marius', 'sergiu']:
    with open(f'dosare/{nume_fisier}.json') as json_file:
        json_var = json.load(json_file)
        json_dataset.extend(json_var)

        # for i in range(len(json_var)):
        #     if len(json_var[i]['annotations']) > 1:
        #         print()
        #         print(f"nume_fisier = {nume_fisier}, id = {json_var[i]['id']}")
        #         for ann in json_var[i]['annotations']:
        #             print(ann)
        #     elif len(json_var[i]['annotations']) == 1:
        #         pass
        #     elif len(json_var[i]['annotations']) == 0:
        #         print('e 0')

output_json = []
id_idx = 0

with IncrementalBar('Tokenizing...', max=len(json_dataset)) as bar:
    for json_entry in json_dataset:
        sentence_dict = {'id': id_idx, 'ner_tags': [], 'ner_ids': [], 'tokens': [], 'space_after': []}

        text = json_entry['data']['ner']

        annotations = json_entry['annotations'][-1]['result'] # -> list (UPDATE: O singura adnotare per exemplu)
        sorted_annotations = sorted(annotations, key=lambda x: x['value']['start'])

        current_idx = 0
        for annotation in sorted_annotations:
            annot = annotation['value']

            start_idx = annot['start']
            if current_idx < start_idx:
                current_text_fragment = text[current_idx:start_idx]
                doc = nlp(current_text_fragment)
                for sentence in doc.sentences:
                    for token in sentence.tokens:
                        not_allowed_chars = set(token.text).difference(ALLOWED_CHARS)
                        if not not_allowed_chars:
                            sentence_dict['tokens'].append(token.text)
                            sentence_dict['ner_tags'].append('O')
                            sentence_dict['ner_ids'].append(0)
                            sentence_dict['space_after'].append(True if (current_idx+token.end_char<len(text) and text[current_idx+token.end_char] == ' ') else False)
                        else:
                            print('Am gasit caractere dubioase')

            current_idx = start_idx
                        
            current_text_fragment = annot['text']
            doc = nlp(current_text_fragment)

            for sentence in doc.sentences:
                flag_first_token = True # Used for special chars, if they are first in token, they will be skipped
                for index, token in enumerate(sentence.tokens):
                    not_allowed_chars = set(token.text).difference(ALLOWED_CHARS)
                    if not not_allowed_chars:
                        sentence_dict['tokens'].append(token.text)
                        if index == 0 or flag_first_token:
                            sentence_dict['ner_tags'].append('B-'+annot['labels'][0])
                            sentence_dict['ner_ids'].append(LABEL_MAPPER[annot['labels'][0]])

                            flag_first_token = False
                        else:
                            sentence_dict['ner_tags'].append('I-'+annot['labels'][0])
                            sentence_dict['ner_ids'].append(LABEL_MAPPER[annot['labels'][0]]+1)
                        sentence_dict['space_after'].append(True if (current_idx+token.end_char<len(text) and text[current_idx+token.end_char] == ' ') else False)
                    elif token.text[0] == '\ufeff' and token.text.replace('\ufeff', ''):
                        sentence_dict['tokens'].append(token.text.replace('\ufeff', ''))
                        if index == 0 or flag_first_token:
                            sentence_dict['ner_tags'].append('B-'+annot['labels'][0])
                            sentence_dict['ner_ids'].append(LABEL_MAPPER[annot['labels'][0]])

                            flag_first_token = False
                        else:
                            sentence_dict['ner_tags'].append('I-'+annot['labels'][0])
                            sentence_dict['ner_ids'].append(LABEL_MAPPER[annot['labels'][0]]+1)
                        sentence_dict['space_after'].append(True if (current_idx+token.end_char<len(text) and text[current_idx+token.end_char] == ' ') else False)
                    else:
                        print('Am gasit caractere dubioase')

            current_idx = annot['end']

        current_text_fragment = text[current_idx:]
        doc = nlp(current_text_fragment)

        for sentence in doc.sentences:
            for token in sentence.tokens:
                not_allowed_chars = set(token.text).difference(ALLOWED_CHARS)
                if not not_allowed_chars:
                    sentence_dict['tokens'].append(token.text)
                    sentence_dict['ner_tags'].append('O')
                    sentence_dict['ner_ids'].append(0)
                    sentence_dict['space_after'].append(True if (current_idx+token.end_char<len(text) and text[current_idx+token.end_char] == ' ') else False)
                else:
                    print('Am gasit caractere dubioase')

        output_json.append(sentence_dict)

        id_idx += 1

        bar.next()

####################################################################
# Pentru a da export                                               #
with open('dosare/export_step1.json', 'w') as output_json_file:    #
    json.dump(output_json, output_json_file)#, indent=4)           #
####################################################################