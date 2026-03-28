
# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import torch

DEVICE_MAP = "auto"

N_CTX = 6144

# How many tokens are processed in parallel, default is 8, set to bigger number.
# Should be between 1 and N_CTX, consider the amount of VRAM in your GPU.
N_BATCH = 512 

# Determines how many layers of the model are offloaded to your Metal GPU, in the most case, 
# set it to 1 is enough for Metal.
# Change this value based on your model and your GPU VRAM pool.
N_GPU_LAYERS = 40 

# Device Type
DEVICE_TYPE_MPS = "mps"
DEVICE_TYPE_CUDA = "cuda"
DEVICE_TYPE_CPU = "cpu"

if torch.backends.mps.is_available():
    DEVICE_TYPE = DEVICE_TYPE_MPS
elif torch.cuda.is_available():
    DEVICE_TYPE = DEVICE_TYPE_CUDA
else:
    DEVICE_TYPE = DEVICE_TYPE_CPU

EMBEDDING_KWARGS = {
    "device": DEVICE_TYPE_CPU,
    "batch_size": N_BATCH
}

ENCODE_KWARG = {
    "normalize_embeddings": True
}

# Embedding settings
DEFAULT_MODEL_NAME = "hkunlp/instructor-large"
DEFAULT_MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
DEFAULT_MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

HUGGINGFACE_EMBEDDING_NAME = "HuggingFaceInstructEmbeddings"
LLAMA_CPP_EMBEDDING_NAME = "llama-2-7b-chat"
OPEN_AI_EMBEDDING_NAME = "OpenAIEmbeddings"
OLLAMA_EMBEDDING_NAME = "OllamaEmbeddings"
GPT4_EMBEDDING_NAME = "GPT4AllEmbeddings"

# Recommended models for embeddings with sub-choices
RECOMMENDED_MODELS = {
    HUGGINGFACE_EMBEDDING_NAME: {
        "hkunlp/instructor-large": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "hkunlp/instructor-xl": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "hkunlp/instructor-base": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        }
    },
    LLAMA_CPP_EMBEDDING_NAME: {
        "llama-2-7b-chat": {
            "model_kwargs": {
                "model_path": "/path/to/llama-2-7b-chat-model",
                "n_gpu_layers": 1,
                "n_batch": 512,
                "n_ctx": 2048,
                "f16_kv": True
            },
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "llama-2-13b-chat": {
            "model_kwargs": {
                "model_path": "/path/to/llama-2-13b-chat-model",
                "n_gpu_layers": 1,
                "n_batch": 512,
                "n_ctx": 2048,
                "f16_kv": True
            },
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "llama-2-70b-chat": {
            "model_kwargs": {
                "model_path": "/path/to/llama-2-70b-chat-model",
                "n_gpu_layers": 1,
                "n_batch": 512,
                "n_ctx": 2048,
                "f16_kv": True
            },
            "encode_kwargs": {"normalize_embeddings": True}
        }
    },
    OPEN_AI_EMBEDDING_NAME: {
        "openai/text-embedding-ada-002": {
            "model_kwargs": {"api_key": "your_openai_api_key"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "openai/text-embedding-babbage-001": {
            "model_kwargs": {"api_key": "your_openai_api_key"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "openai/text-embedding-curie-001": {
            "model_kwargs": {"api_key": "your_openai_api_key"},
            "encode_kwargs": {"normalize_embeddings": True}
        }
    },
    OLLAMA_EMBEDDING_NAME: {
        "ollama-7b": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "ollama-13b": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "ollama-30b": {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        }
    },
    GPT4_EMBEDDING_NAME: {
        "gpt4all-lora-quantized": {
            "model_kwargs": {"model_name": "gpt4all-lora-quantized"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "gpt4all-mpt-7b": {
            "model_kwargs": {"model_name": "gpt4all-mpt-7b"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "gpt4all-j-6b": {
            "model_kwargs": {"model_name": "gpt4all-j-6b"},
            "encode_kwargs": {"normalize_embeddings": True}
        }
    }
}
