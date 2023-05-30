import finetuner

model = finetuner.get_model('/mnt/models/margin-mse-osc-run.zip', device='cpu')


def get_text_sentence_embedding(text: str, normalize: bool = True):
    return finetuner.encode(model=model, data=[text])[0]
