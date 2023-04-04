import finetuner

model = finetuner.get_model('/mnt/models/clip-osc-run.zip', select_model='clip-text', device='cpu')


def get_text_sentence_embedding(text: str, normalize: bool = True):
    bla = finetuner.encode(model=model, data=text)[0]
    print(bla)
    return bla
