from typing import Union, List, Dict, Any
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel

models: Dict[str, BGEM3FlagModel] = {}
model_name = os.getenv("MODEL", "BAAI/bge-m3")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field()
    model: str = Field(
        examples=[model_name],
        default=model_name,
    )
    
    # 添加一个额外的字典字段来接收任意参数
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the encode function"
    )
    
    class Config:
        extra = "allow"  # 允许额外的字段


class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Usage
    object: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    models[model_name] = BGEM3FlagModel(model_name, use_fp16=True)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/embeddings")
async def embedding(item: EmbeddingRequest) -> EmbeddingResponse:
    model: BGEM3FlagModel = models[model_name]
    # 获取所有额外的参数
    encode_kwargs = dict(item.kwargs)
    # 添加请求中的其他额外字段
    encode_kwargs.update({
        k: v for k, v in item.model_dump().items() 
        if k not in {"input", "model", "kwargs"}
    })
    
    if isinstance(item.input, str):
        vectors = model.encode(item.input, **encode_kwargs)
        tokens = len(vectors)
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=vectors.tolist(), index=0, object="embedding")],
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            object="list",
        )
    if isinstance(item.input, list):
        embeddings = []
        tokens = 0
        for index, text_input in enumerate(item.input):
            if not isinstance(text_input, str):
                raise HTTPException(
                    status_code=400,
                    detail="input needs to be an array of strings or a string",
                )
            vectors = model.encode(text_input, **encode_kwargs)
            tokens += len(vectors)
            embeddings.append(
                EmbeddingData(embedding=vectors.tolist(), index=index, object="embedding")
            )
        return EmbeddingResponse(
            data=embeddings,
            model=model_name,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            object="list",
        )
    raise HTTPException(
        status_code=400, detail="input needs to be an array of strings or a string"
    )


@app.get("/")
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
