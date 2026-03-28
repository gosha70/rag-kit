# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
import unittest
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, LlamaCppEmbeddings, OpenAIEmbeddings, OllamaEmbeddings, GPT4AllEmbeddings
from embeddings.model_info import ModelInfo
from embeddings.models_constants import (
    DEFAULT_MODEL_NAME
)

# Load environment variables from .env file
load_dotenv()

class TestModelInfo(unittest.TestCase):

    def setUp(self):
        self.model_info = ModelInfo()

    def test_default_model(self):
        embedding = self.model_info.create_embedding()
        self.assertIsInstance(embedding, HuggingFaceInstructEmbeddings)
        self.assertEqual(embedding.model_name, DEFAULT_MODEL_NAME)

    def test_huggingface_instructor_large(self):
        embedding = self.model_info.create_embedding("hkunlp/instructor-large")
        self.assertIsInstance(embedding, HuggingFaceInstructEmbeddings)
        self.assertEqual(embedding.model_name, "hkunlp/instructor-large")

    def test_llama_cpp(self):
        embedding = self.model_info.create_embedding("llama-2-13b-chat")
        self.assertIsInstance(embedding, LlamaCppEmbeddings)
        self.assertEqual(embedding.model_name, "llama-2-13b-chat")

"""
    def test_openai_embeddings(self):
        embedding = self.model_info.create_embedding("openai/text-embedding-ada-002")
        self.assertIsInstance(embedding, OpenAIEmbeddings)
        self.assertEqual(embedding.model_name, "openai/text-embedding-ada-002")

    def test_ollama_embeddings(self):
        embedding = self.model_info.create_embedding("ollama-13b")
        self.assertIsInstance(embedding, OllamaEmbeddings)
        self.assertEqual(embedding.model_name, "ollama-13b")

    def test_gpt4all_embeddings(self):
        embedding = self.model_info.create_embedding("gpt4all-mpt-7b")
        self.assertIsInstance(embedding, GPT4AllEmbeddings)
        self.assertEqual(embedding.model_name, "gpt4all-mpt-7b")
"""
if __name__ == "__main__":
    unittest.main()
