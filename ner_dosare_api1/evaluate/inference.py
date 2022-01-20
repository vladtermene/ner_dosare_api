from model import TransformerModel


if __name__ == "__main__":
    # this is how to run
    device = "cpu" # or "cuda"
    model = TransformerModel.load(folder='trained_model/dosare_first_model')
    model.set_device(device)

    sentence = "Degeratu Dorin Titi prin S.C.A Cioana si Asociatii"
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

    for i in range(len(tokens)):
        print(f"{tokens[i]: >16} = {tags[i]: <12}")
