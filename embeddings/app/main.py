from fastapi import FastAPI
from clip import router as router_clip
from minilm import router as router_minilm
from finetuned import sbert_router, clip_router

app = FastAPI()

app.include_router(router_clip.router, prefix="/clip")
app.include_router(router_minilm.router, prefix="/minilm")
app.include_router(clip_router, prefix="/finetunedclip")
app.include_router(sbert_router, prefix="/finetunedsbert")


