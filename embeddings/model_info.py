
# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, LlamaCppEmbeddings, OllamaEmbeddings, GPT4AllEmbeddings
from .models_constants import (
    DEFAULT_MODEL_BASENAME,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    DEVICE_TYPE_CPU,
    EMBEDDING_KWARGS,
    ENCODE_KWARG,
    RECOMMENDED_MODELS,
    HUGGINGFACE_EMBEDDING_NAME,
    LLAMA_CPP_EMBEDDING_NAME,
    OPEN_AI_EMBEDDING_NAME,
    OLLAMA_EMBEDDING_NAME,
    GPT4_EMBEDDING_NAME
)

"""
ModelInfo class defines the setting for creating and training LLM used in the retrieval framework.
    - model_name (str) 
    - model_id (str) 
    - model_basename (str) 
    - device_type (str): 'cpu', 'cuda', or 'mps'

See: https://huggingface.co/docs/transformers/model_doc/auto    
"""
class ModelInfo:
    def __init__(self):
        self._model_name = DEFAULT_MODEL_NAME
        self._model_id = DEFAULT_MODEL_ID
        self._model_basename = DEFAULT_MODEL_BASENAME
        self._device_type = DEVICE_TYPE_CPU  # 'cpu', 'cuda', or 'mps'

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    @property
    def model_basename(self):
        return self._model_basename

    @model_basename.setter
    def model_basename(self, value):
        self._model_basename = value

    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, value):
        self._device_type = value

    def __str__(self):
        return (f"ModelInfo(model_name='{self._model_name}', "
                f"model_id='{self._model_id}', "
                f"model_basename='{self._model_basename}', "
                f"device_type='{self._device_type}')")
    
    def create_embedding(self, model_name=None):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        embedding_class, model_config = self.get_embedding_class_and_config(model_name)

        if embedding_class == OpenAIEmbeddings:
            # Extract api_key from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            return embedding_class(api_key=api_key)
        else:
            return embedding_class(
                model_name=model_name,
                model_kwargs=model_config.get("model_kwargs", EMBEDDING_KWARGS),
                encode_kwargs=model_config.get("encode_kwargs", ENCODE_KWARG)
            )


    def get_embedding_class_and_config(self, model_name):
        for embedding_name, models in RECOMMENDED_MODELS.items():
            if model_name in models:
                if embedding_name == HUGGINGFACE_EMBEDDING_NAME:
                    return HuggingFaceInstructEmbeddings, models[model_name]
                elif embedding_name == LLAMA_CPP_EMBEDDING_NAME:
                    return LlamaCppEmbeddings, models[model_name]
                elif embedding_name == OPEN_AI_EMBEDDING_NAME:
                    return OpenAIEmbeddings, models[model_name]
                elif embedding_name == OLLAMA_EMBEDDING_NAME:
                    return OllamaEmbeddings, models[model_name]
                elif embedding_name == GPT4_EMBEDDING_NAME:
                    return GPT4AllEmbeddings, models[model_name]
        # Default to HuggingFaceInstructEmbeddings if no match found
        return HuggingFaceInstructEmbeddings, RECOMMENDED_MODELS[HUGGINGFACE_EMBEDDING_NAME][DEFAULT_MODEL_NAME]

    def embedding_class(self, model_name=None):
        embedding_cls, _ = self.get_embedding_class_and_config(model_name or DEFAULT_MODEL_NAME)
        return f"{embedding_cls.__module__}.{embedding_cls.__name__}"
   