# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import os
from datetime import datetime
import json
import zipfile
import tempfile
from typing import List
from langchain_core.documents import Document

from rag.embeddings.chroma.embeddings_chroma_constants import get_elapse_time_message, KWARGS_PARAM_NAME, PAGE_CONTENT_PARAM_NAME, METADATA_PARAM_NAME
from unstructured.file_type import FileType
from unstructured.file_loader_query import FileLoaderQuery
from unstructured.document_splitter import DocumentSplitter
from unstructured.converters.base_file_converter import BaseFileConverter

class DocumentsStoreService:
    """
    Finds files and splits them into unstructured text.
    """
    def __init__(self, document_splitter: DocumentSplitter, logging):
        self.logging = logging
        self.document_splitter = document_splitter
        
    def load_supported_documents(self, dir_path: str) -> List[Document]:
        """
        Finds and loads all files corresponding to supported file types and counts them.

        Parameters:
        - dir_path (str): The root directory where the search for documents is performed

        Returns:
        - (List[Document]): unstructured document splits
        """
        self.logging.info("Loading files with supported extensions...")   
        files_by_type = {file_type: [] for file_type in FileType} 
        for root, _, files in os.walk(dir_path):
            for file in files:
                for file_type in FileType:
                    if file.endswith(file_type.get_extension()):
                        file_path = os.path.join(root, file)
                        files_by_type[file_type].append(file_path)

        split_docs = [] 
        file_type_counts = {file_type: 0 for file_type in FileType} 
        for file_type, files in files_by_type.items():
            if len(files) > 0: 
                text_splitter = BaseFileConverter.get_text_splitter(file_type)
                if text_splitter is None:
                    self.logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                    continue
                for file_path in files:
                    file_splits = self.document_splitter.load_split_file(text_splitter, file_type, file_path)
                    split_docs.extend(file_splits)
                    file_type_counts[file_type] += 1

        self.logging.info(f"Total document splits: {len(split_docs)}")  
        # Log the count of each file type found
        for file_type, count in file_type_counts.items():
            if count > 0:
                self.logging.info(f"Found {count} '{file_type.value}' files.")
    
        return split_docs

    def load_documents(self, dir_path: str, file_loader_query: FileLoaderQuery) -> List[Document]:
        """
        Loads files in the specified directory into unstructured document splits.

        Parameters:
        - dir_path (str): The root directory where the search for documents is performed
        - file_loader_query (FileLoaderQuery): The FileLoaderQuery holds the search criteria for files to laod and analyze

        Returns:
        - (List[Document]): unstructured document splits

        See: https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html
        """
        try:
            split_docs = []
            # Iterate over the file types and their patterns
            for file_type in file_loader_query.patterns:
                text_splitter = BaseFileConverter.get_text_splitter(file_type)
                if text_splitter is None:
                    self.logging.warning(f"Cannot find (TextSplitter) for {file_type.get_extension()}")
                    continue
                patterns = file_loader_query.get_patterns(file_type)
                for pattern in patterns:
                    if file_type is None:
                        raise ValueError(f"Got the unsupported field type '{file_type.get_extension()}'")
                    file_type_docs = self.document_splitter.load_and_split(dir_path, text_splitter, file_type, pattern)                    
                    split_docs.extend(file_type_docs)

            self.logging.info(f"Total number of unstructured document splits: {len(split_docs)}")

            return split_docs 
        except Exception as error:
            self.logging.error(f"Failed to process documents with extensions '{file_loader_query}' found in the path '{dir_path}': {str(error)}", exc_info=True)

            return None

    def save_splits_to_disk(self, split_docs, output_dir: str = None):  
        """
        Saves each split document as a separate file in the specified output directory.
        If output_dir is None, creates a new directory in the current directory.

        Parameters:
        - split_docs (List[Document]): List of split document objects
        - output_dir (str): The directory where the split documents will be saved. Defaults to None.

        Returns: the directory where doocuments are saved
        """
        if output_dir is None:
            # Create a new directory with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.getcwd(), f"split_docs_{timestamp}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, doc in enumerate(split_docs):
            doc_json = doc.to_json()  # Convert Document object to Json
            with open(os.path.join(output_dir, f"doc_split_{i}.json"), "w", encoding="utf-8") as file:
                json.dump(doc_json, file, indent=4)

        return output_dir 

    def load_document_split(self, split_file) -> Document:
        """
        Craete Document from the specified JSON file.

        Parameters:
        - split_file (File): the JSON file sstoring a single unstructured document split

        Returns (Document)
        """
        try:
            # Read and process the content as JSON
            json_content = split_file.read()

            # Parse the JSON content
            data = json.loads(json_content)

            # Access the "page_content" field
            page_content = data[KWARGS_PARAM_NAME][PAGE_CONTENT_PARAM_NAME]

            # Access the "metadata" field
            metadata = data[KWARGS_PARAM_NAME][METADATA_PARAM_NAME]

            # Transform the data into a langchain_core.documents.Document
            # Assuming the JSON structure fits the Document's requirements
            return Document(page_content=page_content, metadata=metadata)
        except Exception as error:
            self.logging.error(f"File {split_file} is not a valid JSON: {str(error)}")

        return None

    def load_zip_with_splits(self, zip_file, unzip_folder=None) -> str:
        """
        Extracts the specified zip with unstructured document splits to the specified folder.

        Parameters:
        - zip_file (List[Document]): List of split document objects
        - output_dir (str): The directory where the split documents will be saved. Defaults to None.

        Returns: the directory where doocuments are saved
        """    
        if unzip_folder is None:
            curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_folder = f"unzip_{curr_time}" 
            unzip_folder = tempfile.mkdtemp(prefix=new_folder)
        else:
            # Create the directory if it does not exist
            os.makedirs(unzip_folder, exist_ok=True)

        # Open the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all the contents into the directory
            zip_ref.extractall(unzip_folder)

        return unzip_folder    
