# BGEM3 Flag Embeddings Docker Deployment

OpenAI compatible embedding API that uses Sentence Transformer for embeddings

Container Image: `ghcr.io/betterandbetterii/jina-embeddings-docker`

### Install (Docker)
Run the API locally using Docker:
```bash
docker run --runtime nvidia --gpus all -e MODEL=jinaai/jina-embeddings-v3 -p 8080:8080 -v /srv/jinaai/models/cache:/root/.cache -d ghcr.io/betterandbetterii/stapi:main
```

## Usage
After you've installed STAPI,
you can visit the API docs on [http://localhost:8080/docs](http://localhost:8080/docs)

You can also use CURL to get embeddings:
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text string goes here",
    "model": "all-MiniLM-L6-v2"
  }'
```

Even the OpenAI Python client can be used to get embeddings:
```python
import openai
openai.api_base = "http://localhost:8080/v1"
openai.api_key = "this isn't used but openai client requires it"
model = "all-MiniLM-L6-v2"
embedding = openai.Embedding.create(input="Some text", model=model)["data"][0]["embedding"]
print(embedding)
```

## Supported Models
Any model that's supported by Sentence Transformers should also work as-is
with STAPI.
Here is a list of [pre-trained models](https://www.sbert.net/docs/pretrained_models.html) available with Sentence Transformers.

By default the `all-MiniLM-L6-v2` model is used and preloaded on startup. You
can preload any supported model by setting the `MODEL` environment variable.

For example, if you want to preload the `multi-qa-MiniLM-L6-cos-v1`, you
could tweak the `docker run` command like this:
```bash
docker run -e MODEL=multi-qa-MiniLM-L6-cos-v1  -p 8080:8080 -d \
  ghcr.io/substratusai/sentence-transformers-api
```

Note that STAPI will only serve the model that it is preloaded with. You
should create another instance of STAPI to serve another model. The `model`
parameter as part of the request body is simply ignored.


## Integrations
It's easy to utilize the embedding server with various other tools because
the API is compatible with the OpenAI Embedding API.

### Weaviate
You can use the Weaviate text2vec-openai module and use the
STAPI OpenAI compatible endpoint.

In your Weaviate Schema
use the following module config, assuming STAPI endpoint
is available at `http://stapi:8080`:
```
  "vectorizer": "text2vec-openai",
  "moduleConfig": {
    "text2vec-openai": {
      "model": "davinci",
      "baseURL": "http://stapi:8080"
    }
  }
```
For the OpenAI API key you can use any key, it won't be checked.

Read the [STAPI Weaviate Guide](https://github.com/substratusai/stapi/tree/main/weaviate) for more details.

## Creators
Feel free to contact any of us:
* [Sam Stoelinga aka Samos123](https://www.linkedin.com/in/samstoelinga/)
* [Nick Stogner](https://www.linkedin.com/in/nstogner/)
