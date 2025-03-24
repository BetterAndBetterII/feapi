from typing import Union, List, Dict, Any, Optional, Literal
from contextlib import asynccontextmanager
import os
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator, root_validator
from FlagEmbedding import BGEM3FlagModel

models: Dict[str, BGEM3FlagModel] = {}
model_name = os.getenv("MODEL", "BAAI/bge-m3")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field()
    model: str = Field(
        examples=[model_name],
        default=model_name,
    )
    
    # 添加BGR-M3特定参数
    return_dense: Optional[bool] = Field(
        default=True,
        description="是否返回密集向量表示"
    )
    return_sparse: Optional[bool] = Field(
        default=False,
        description="是否返回稀疏向量表示(词汇权重)"
    )
    return_colbert_vecs: Optional[bool] = Field(
        default=False,
        description="是否返回ColBERT向量表示"
    )
    
    # 添加一个额外的字典字段来接收任意参数
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the encode function"
    )
    
    class Config:
        extra = "allow"  # 允许额外的字段
        
    @root_validator
    def validate_return_flags(cls, values):
        return_dense = values.get("return_dense")
        return_sparse = values.get("return_sparse")
        return_colbert_vecs = values.get("return_colbert_vecs")
        
        # 检查是否只有一个标志为True
        true_count = sum(bool(flag) for flag in [return_dense, return_sparse, return_colbert_vecs] if flag is not None)
        
        if true_count > 1:
            raise ValueError("只能设置一个返回类型为True: return_dense, return_sparse, return_colbert_vecs")
        
        # 如果都是False或None，默认使用dense
        if true_count == 0:
            values["return_dense"] = True
            
        return values


class EmbeddingData(BaseModel):
    embedding: Union[List[float], Dict[str, float], List[List[float]]]
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
    
    # 添加BGR-M3特定参数
    encode_kwargs.update({
        "return_dense": item.return_dense,
        "return_sparse": item.return_sparse,
        "return_colbert_vecs": item.return_colbert_vecs
    })
    
    # 添加请求中的其他额外字段
    encode_kwargs.update({
        k: v for k, v in item.model_dump().items() 
        if k not in {"input", "model", "kwargs", "return_dense", "return_sparse", "return_colbert_vecs"}
    })
    
    if isinstance(item.input, str):
        result = model.encode(item.input, **encode_kwargs)
        
        # 根据返回类型提取相应的向量
        if item.return_dense:
            vectors = result["dense_vecs"]
            tokens = len(vectors)
            vector_data = vectors.tolist()
            vec_type = "dense_embedding"
        elif item.return_sparse:
            vectors = result["lexical_weights"]
            tokens = len(next(iter(vectors)))  # 估算token数量
            vector_data = vectors
            vec_type = "sparse_embedding"
        elif item.return_colbert_vecs:
            vectors = result["colbert_vecs"]
            tokens = len(vectors)  # 估算token数量
            vector_data = [v.tolist() for v in vectors]
            vec_type = "colbert_embedding"
        
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=vector_data, index=0, object=vec_type)],
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
                
            result = model.encode(text_input, **encode_kwargs)
            
            # 根据返回类型提取相应的向量
            if item.return_dense:
                vectors = result["dense_vecs"]
                cur_tokens = len(vectors)
                vector_data = vectors.tolist()
                vec_type = "dense_embedding"
            elif item.return_sparse:
                vectors = result["lexical_weights"]
                cur_tokens = len(next(iter(vectors)))  # 估算token数量
                vector_data = vectors
                vec_type = "sparse_embedding"
            elif item.return_colbert_vecs:
                vectors = result["colbert_vecs"]
                cur_tokens = len(vectors)  # 估算token数量
                vector_data = [v.tolist() for v in vectors]
                vec_type = "colbert_embedding"
                
            tokens += cur_tokens
            
            embeddings.append(
                EmbeddingData(embedding=vector_data, index=index, object=vec_type)
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
