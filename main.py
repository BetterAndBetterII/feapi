from typing import Union, List, Dict, Any, Optional, Literal
from contextlib import asynccontextmanager
import os
import numpy as np
import base64

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
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
    
    # OpenAI兼容性参数
    dimensions: Optional[int] = Field(
        default=None,
        description="输出向量的维度"
    )
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        default="float",
        description="输出向量的编码格式：float或base64"
    )
    
    # 添加一个额外的字典字段来接收任意参数
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the encode function"
    )
    
    class Config:
        extra = "allow"  # 允许额外的字段
        
    @model_validator(mode='after')
    def validate_return_flags(self):
        return_dense = self.return_dense
        return_sparse = self.return_sparse
        return_colbert_vecs = self.return_colbert_vecs
        
        # 检查是否只有一个标志为True
        true_count = sum(bool(flag) for flag in [return_dense, return_sparse, return_colbert_vecs] if flag is not None)
        
        if true_count > 1:
            raise ValueError("只能设置一个返回类型为True: return_dense, return_sparse, return_colbert_vecs")
        
        # 如果都是False或None，默认使用dense
        if true_count == 0:
            self.return_dense = True
            
        return self


class EmbeddingData(BaseModel):
    embedding: Union[List[float], Dict[str, float], List[List[float]], str]
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


def process_vector_data(vector_data, dimensions=None, encoding_format="float"):
    """处理向量数据，包括维度截断和编码格式转换"""
    # 处理维度
    if dimensions is not None:
        if isinstance(vector_data, list):
            if isinstance(vector_data[0], list):
                # ColBERT向量的情况
                return process_vector_data([vec[:dimensions] for vec in vector_data], 
                                          encoding_format=encoding_format)
            elif isinstance(vector_data[0], (int, float)):
                # 密集向量的情况
                if len(vector_data) > dimensions:
                    vector_data = vector_data[:dimensions]
                elif len(vector_data) < dimensions:
                    # 填充到指定维度
                    vector_data = vector_data + [0.0] * (dimensions - len(vector_data))
        elif isinstance(vector_data, dict):
            # 稀疏向量不支持维度调整
            pass
    
    # 处理编码格式
    if encoding_format == "base64" and not isinstance(vector_data, dict):
        # 只对密集向量和ColBERT向量进行base64编码
        if isinstance(vector_data, list):
            if isinstance(vector_data[0], list):
                # ColBERT向量的情况 - 转为浮点数组
                import numpy as np
                flat_vector = np.array(vector_data).flatten().astype(np.float32).tobytes()
                return base64.b64encode(flat_vector).decode('utf-8')
            else:
                # 密集向量的情况
                import numpy as np
                vector_bytes = np.array(vector_data).astype(np.float32).tobytes()
                return base64.b64encode(vector_bytes).decode('utf-8')
    
    return vector_data


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
    
    # 获取OpenAI兼容性参数但不传递给encode
    dimensions = item.dimensions
    encoding_format = item.encoding_format or "float"
    
    # 移除不兼容的参数，避免传递给encode
    incompatible_params = ["dimensions", "encoding_format"]
    for param in incompatible_params:
        if param in encode_kwargs:
            del encode_kwargs[param]
    
    # 添加请求中的其他额外字段
    encode_kwargs.update({
        k: v for k, v in item.model_dump().items() 
        if k not in {"input", "model", "kwargs", "return_dense", "return_sparse", 
                     "return_colbert_vecs", "dimensions", "encoding_format"}
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
            # 增加鲁棒性：检查词汇权重字典是否为空
            if not vectors:
                tokens = 1  # 当词汇权重为空时设置默认token数
            else:
                try:
                    tokens = len(next(iter(vectors)))  # 尝试估算token数量
                except StopIteration:
                    tokens = 1  # 异常情况下设置默认token数
            vector_data = vectors
            vec_type = "sparse_embedding"
        elif item.return_colbert_vecs:
            vectors = result["colbert_vecs"]
            tokens = len(vectors)  # 估算token数量
            vector_data = [v.tolist() for v in vectors]
            vec_type = "colbert_embedding"
        
        # 处理向量数据
        processed_data = process_vector_data(vector_data, dimensions, encoding_format)
        
        return EmbeddingResponse(
            data=[EmbeddingData(embedding=processed_data, index=0, object=vec_type)],
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
                # 增加鲁棒性：检查词汇权重字典是否为空
                if not vectors:
                    cur_tokens = 1  # 当词汇权重为空时设置默认token数
                else:
                    try:
                        cur_tokens = len(next(iter(vectors)))  # 尝试估算token数量
                    except StopIteration:
                        cur_tokens = 1  # 异常情况下设置默认token数
                vector_data = vectors
                vec_type = "sparse_embedding"
            elif item.return_colbert_vecs:
                vectors = result["colbert_vecs"]
                cur_tokens = len(vectors)  # 估算token数量
                vector_data = [v.tolist() for v in vectors]
                vec_type = "colbert_embedding"
                
            tokens += cur_tokens
            
            # 处理向量数据
            processed_data = process_vector_data(vector_data, dimensions, encoding_format)
            
            embeddings.append(
                EmbeddingData(embedding=processed_data, index=index, object=vec_type)
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
