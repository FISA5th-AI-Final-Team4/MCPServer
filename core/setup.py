from fastapi import FastAPI

from core.qdrant_upsert import upsert_card_desc_data

def lifespan(app: FastAPI):
    upsert_card_desc_data()
    yield