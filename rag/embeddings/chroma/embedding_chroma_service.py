# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import argparse
import asyncio
import logging
import time
import os
from datetime import datetime
from langchain_chroma import Chroma

from rag.embeddings.chroma.embeddings_chroma_constants import CHROMA_SETTINGS, DEFAULT_COLLECTION_NAME, BATCH_SIZE, get_elapse_time_message

from embeddings.model_info import ModelInfo
from embeddings.models_constants import DEFAULT_MODEL_NAME

from vectorstores.documents_store_service import DocumentsStoreService 

from unstructured.file_loader_query import FileLoaderQuery
from unstructured.document_splitter import DocumentSplitter

class EmbeddingService:
    """
    Finds files and splits them into unstructured text.
    """
    def __init__(self, documents_store_service: DocumentsStoreService, logging):
        self.logging = logging
        self.documents_store_service = documents_store_service

    def create_manifest(self, collection_name, model_name, persist_directory):
        if persist_directory is None:
            return False
        
        if collection_name is None:
            collection_name = DEFAULT_COLLECTION_NAME

        if model_name is None:
            model_name = DEFAULT_MODEL_NAME    
    
        meta_inf_path = os.path.join(persist_directory, 'META-INF')
        manifest_file_path = os.path.join(meta_inf_path, 'MANIFEST.MF')
        
        # Get current UTC time and format it
        current_utc_datetime = datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' 

        # Manifest content
        manifest_content = f"""
        Manifest-Version: 1.0\nCreated-On: {current_utc_datetime}
        Created-By: EGOGE (https://github.com/gosha70/document-assistant)
        Collectio-Name: {collection_name}
        Embedding-Class: {ModelInfo().embedding_class(model_name)}
        Embedding-Model-Name: {model_name}
        """

        if not os.path.exists(meta_inf_path):
            os.makedirs(meta_inf_path)
            self.logging.info(f"Created directory: {meta_inf_path}")

        with open(manifest_file_path, 'w') as file:
            file.write(manifest_content)
            self.logging.info(f"Created MANIFEST.MF at: {manifest_file_path}")

        return True    


    def on_task_completion(self, task_id, completed_tasks):
        # Update the list of completed tasks
        completed_tasks.append(task_id)
        self.logging.info(f"The async task #{task_id} just finished. Total finished tasks: {len(completed_tasks)}")

    async def wait_for_tasks(self, docs_db, async_tasks, completed_tasks) -> Chroma:
        """
        Waits for the specified async tasks to finish, then saves the vectorstoore.

        Parameters:
        - docs_db (Chroma): the vectorstore
        - async_tasks (List): the collection of isssued async tasks 
        - completed_tasks (List): the collection of finished async tasks 

        Returns:
        - (Chroma): the embedding vectorstore
        """
        # Wait for all scheduled tasks to complete
        if async_tasks:
            in_progress_tasks = len(async_tasks) - len(completed_tasks)
            self.logging.info(f"Waiting for {in_progress_tasks} async tasks to finish ...")
            await asyncio.gather(*async_tasks)
            self.logging.info(f"All {len(async_tasks)} async tasks finished.")

        # Save the Chroma database after processing all chunks
        if docs_db is not None:
            self.logging.info("Saving the vectorstore ...")
            docs_db.persist()

        return docs_db

    async def process_chunks(self, docs_db, documents, embedding, collection_name, persist_directory, async_tasks, completed_tasks, task_id) -> Chroma:
        """
        Processes the specified chunk of (Documents).

        Parameters:
        - docs_db (Chroma): if the specified vectorstore is None, - syncronously creates one with the specified documents;
                            otherwise, - asyncronously adds the specified documents to the vectorstore. 
        - documents (List[Document]): The unstructured document splits
        - embedding: The LLM used as embedding to process documents
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding database; 
                                    if it is not specified, (Chroma) is not persisted.
        - async_tasks (List): the collection of isssued async tasks
        - completed_tasks (List): the collection of finished async tasks
        - task_id (int): the async task id

        Returns:
        - (Chroma): the embedding vectorstore
        """
        if not docs_db:
            self.logging.info(f"Creating the embedding vectorstore with {embedding} for {len(documents)} document splits ...")    
            if collection_name is None:
                collection_name = DEFAULT_COLLECTION_NAME

            docs_db = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=embedding, 
                persist_directory=persist_directory
            )
            self.logging.info("Finished the creation of embedding vectorstore.")    
        else:
            self.logging.info(f"The async task #{task_id} is updating the embedding vectorstore with {len(documents)} document splits ...")
            async_task = asyncio.create_task(docs_db.aadd_documents(documents=documents))
            async_task.add_done_callback(lambda future, i=task_id: self.on_task_completion(i, completed_tasks))       
            async_tasks.append(async_task)

        return docs_db

    def add_file_content_to_db(self, docs_db: Chroma, file_name: str):
        """
        Processes the single file.

        Parameters:
        - docs_db (Chroma): if the specified vectorstore is None, - syncronously creates one with the specified documents;
                            otherwise, - asyncronously adds the specified documents to the vectorstore. 
        - document_splitter (DocumentSplitter): the document splitter                     
        - file_name (str): the file name
        """
        documents = self.documents_store_service.document_splitter.process_file(file_path=file_name)  
        if documents is not None:  
            self.logging.info(f"Updating the embedding vectorstore with {len(documents)} document splits ...")
            ids = docs_db.add_documents(documents=documents) 
            if ids:
                self.logging.info(f"Saving the vectorstore with new document ids: {ids}")
                docs_db.persist()
            
    @staticmethod
    def adjust_batch_size(batch_size, items_count):
        max_thread = items_count / batch_size
        # Limit to 10 concurrent thread (the first batch is processed synchronously)
        if max_thread > 11:
            batch_size = items_count / 11
        return int(batch_size)    

    async def process_splits_in_chunks(self, embedding, documents, chunk_size, collection_name, persist_directory) -> Chroma:
        """
        Add the specified (Documents) in chunks to a new (Chroma) vectorstore.

        Parameters:
        - embedding: The LLM used as embedding to process documents
        - documents (List[Document]): The unstructured document splits
        - chunk_size (int): The size of each batch/chunk added to a vectorstore
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding database; 
                                    if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore
        """
        docs_db = None
        total_count = len(documents)
        async_tasks = []
        completed_tasks = []

        documents_count = len(documents)
        chunk_size = EmbeddingService.adjust_batch_size(batch_size=chunk_size, items_count=documents_count)

        # Process in chunks
        for i in range(0, documents_count, chunk_size):
            document_chunk = documents[i:i + chunk_size]
            batch_id = f"{i // chunk_size + 1}/{total_count // chunk_size + 1}"
            self.logging.info(f"Processing the batch {batch_id}: {len(document_chunk)} documents")

            if document_chunk:
                docs_db = await self.process_chunks(docs_db, document_chunk, embedding, collection_name, persist_directory, async_tasks, completed_tasks, i)

        return await self.wait_for_tasks(docs_db, async_tasks, completed_tasks)

    async def process_files_in_chunks(self, embedding, file_paths, chunk_size, collection_name, persist_directory) -> Chroma:
        """
        Add the specified (Documents) in chunks to a new (Chroma) vectorstore.

        Parameters:
        - embedding: The LLM used as embedding to process documents
        - file_paths (List[str]): Files with JSON storing unstructured document splits
        - chunk_size (int): The size of each batch/chunk added to a vectorstore
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding database; 
                                    if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore
        """
        docs_db = None
        total_count = len(file_paths)
        async_tasks = []
        completed_tasks = []
        # Process in chunks
        task_id = 0

        files_count = len(file_paths)
        chunk_size = EmbeddingService.adjust_batch_size(batch_size=chunk_size, items_count=files_count)

        for i in range(0, files_count, chunk_size):
            file_chunk = file_paths[i:i + chunk_size]
            batch_id = f"{i // chunk_size + 1}/{total_count // chunk_size + 1}"
            self.logging.info(f"Processing the batch {batch_id}: {len(file_chunk)} documents")
            if file_chunk:
                documents = [self.documents_store_service.load_document_split(open(file_path, 'r')) for file_path in file_chunk if file_path]
                self.logging.info(f"Starting the task {task_id} ...")
                docs_db = await self.process_chunks(docs_db, documents, embedding, collection_name, persist_directory, async_tasks, completed_tasks, task_id)
                task_id = task_id + 1

        return await self.wait_for_tasks(docs_db, async_tasks, completed_tasks)

    async def create_embedding_database(self, documents, model_name, chunk_size, collection_name, persist_directory) -> Chroma:
        """
        Creates a (Chroma) embedding vectorstore which stores processed unstructured document splits.

        Parameters:
        - documents (List[Document]): The unstructured document splits
        - model_name (str): The embedding model name
        - chunk_size (int): The size of each batch/chunk added to a vectorstore
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding vectorstore; 
                                if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore
        """
        if documents is None or len(documents) == 0:
            self.logging.warning(f"Cannot create an embedding database from empty list of documents!")  
            return None  

        embedding = ModelInfo().create_embedding(model_name=model_name)

        return await self.process_splits_in_chunks(
            embedding=embedding, 
            documents=documents, 
            chunk_size=chunk_size,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

    async def create_embedding_database_from_splits(self, splits_directory, model_name, chunk_size, collection_name, persist_directory) -> Chroma:
        """
        Creates a (Chroma) embedding vectorstore from unstructured document splits.

        Parameters:
        - splits_directory (str): The full path to directory with unstructured documents
        - model_name (str): The embedding model name
        - chunk_size (int): The size of each batch/chunk added to a vectorstore
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding vectorstore; 
                                if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore
        """    
        self.logging.info(f"Creating the embedding vectorstore from splits in the directory: '{splits_directory}' ...")    
        # Collect all file paths
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(splits_directory):
            for file_name in filenames:
                full_path = os.path.join(dirpath, file_name)
                file_paths.append(full_path)

        self.logging.info(f"Found {len(file_paths)} files")

        embedding = ModelInfo().create_embedding(model_name=model_name)

        return await self.process_files_in_chunks(
            embedding=embedding, 
            file_paths=file_paths, 
            chunk_size=chunk_size,
            collection_name=collection_name,
            persist_directory=persist_directory
        )   

    async def create_embedding_database_from_zip(self, zip_file, model_name, chunk_size, collection_name, persist_directory) -> Chroma:
        """
        Creates a (Chroma) embedding vectorstore from the spcified zip file which stores processed unstructured document splits.

        Parameters:
        - zip_file (str): The full path to the zip file
        - model_name (str): The embedding model name
        - chunk_size (int): The size of each batch/chunk added to a vectorstore
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding vectorstore;
                                if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore
        """
        self.logging.info(f"Creating the embedding vectorstore from the zip file: '{zip_file}' ...")
        unzip_folder = self.documents_store_service.load_zip_with_splits(zip_file=zip_file)

        if unzip_folder is None:
            self.logging.warning(f"Cannot create an embedding database from empty zip: {zip_file}")
            return None

        file_paths = []
        for dirpath, dirnames, filenames in os.walk(unzip_folder):
            for file_name in filenames:
                full_path = os.path.join(dirpath, file_name)
                file_paths.append(full_path)

        embedding = ModelInfo().create_embedding(model_name=model_name)

        return await self.process_files_in_chunks(
            embedding=embedding,
            file_paths=file_paths,
            chunk_size=chunk_size,
            collection_name=collection_name,
            persist_directory=persist_directory
        )


    def create_vector_store(self, documents, model_name, collection_name, persist_directory) -> Chroma:
        """
        Creates a (Chroma) embedding vectorstore which stores processed unstructured document splits
        associates with the specified file types.

        Parameters:
        - documents (List[Document]): The documents to store in the vectorstore
        - model_name (str): The embedding model name; if it is not specified, it is set to "BAAI/bge-small-en"
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding vectorstore; 
                                if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore if documents are found and processed; otherwise - None.
        """
        if documents:
            # Create embeddings and database
            return asyncio.run(self.create_embedding_database(
                documents=documents,
                model_name=model_name, 
                chunk_size=BATCH_SIZE,
                collection_name=collection_name,
                persist_directory=persist_directory
            ))
        
        return None

    def load_vector_store(self, model_name, collection_name, persist_directory) -> Chroma:
        """
        Load the Chroma for the vectorstore persisted in the specified directory.

        Parameters:
        - model_name (str): The embedding model name
        - collection_name (str): the vectorstore collection name
        - persist_directory (str): The optional file path to store the embedding vectorstore; 
                                if it is not specified, (Chroma) is not persisted.

        Returns:
        - (Chroma): the embedding vectorstore if documents are found and processed; otherwise - None.
        """
        self.logging.info(f"Creating embeddings with '{model_name}' LLM ...")
        
        embedding = ModelInfo().create_embedding(model_name=model_name)

        if collection_name is None:
            collection_name = DEFAULT_COLLECTION_NAME

        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embedding,
            client_settings=CHROMA_SETTINGS,
        )


if __name__ == "__main__":
    """
    Main function to load documents, split them, and create a vectorstore with this splits of found documents.
    """
    # Set the logging level to INFO
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the embedding database.")

    # Add the arguments
    parser.add_argument(
        '--dir_path',
        type=str,
        help='Root directory where to look for documents.',
        default="."
    )
    parser.add_argument(
        '--zip_file',
        type=str,
        help='(Optional) The zip file contains unstructured document splits. If this parameter is specified, --dir_path, --file_types, --file_patterns are ignored.',
        default=None
    )
    parser.add_argument(
        '--splits_directory',
        type=str,
        help='(Optional) The directory with unstructured document splits for createing the vectorestore. If this parameter is specified, --dir_path, --file_types, --file_patterns are ignored.',
        default=None
    )
    parser.add_argument(
        '--file_types',
        type=str,
        nargs='+',
        help='(Optional) List of file extensions (without the dot) to find in the specified directory; if the argument is missing, then all supported files are searched.',
        default=None
    )
    parser.add_argument(
        '--file_patterns',
        nargs='+',
        help='Name patterns for each file type; for example: --file_patterns "java:**/*Function* html:**/*"',
        default=[]
    )
    parser.add_argument(
        '--persist_directory',
        type=str,
        help='(Optional) The path to the directory where the vectorsstore will be persisted.',
        default=None
    )
    parser.add_argument(
        '--model_name',
        type=str,
        help='The name of embedding model for analyzing unstructured document splits.',
        default=None
    )
    parser.add_argument(
        '--collection_name',
        type=str,
        help='The name of vectorsstore.',
        default=None
    )
    parser.add_argument(
        '--test_question',
        type=str,
        help='(Optional) Indicates if the created vectorsstore should be tested.',
        default=None
    )

    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Creating the vectorsstore the arguments: {args}")
    document_splitter = DocumentSplitter(logging)
    documents_store_service = DocumentsStoreService(document_splitter=document_splitter, logging=logging)
    embedding_service = EmbeddingService(documents_store_service=documents_store_service, logging=logging)

    # Call the create_vector_store function
    start_time = time.time()

    if args.zip_file:
        docs_db = asyncio.run(embedding_service.create_embedding_database_from_zip(
            zip_file=args.zip_file,
            model_name=args.model_name,
            chunk_size=BATCH_SIZE,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory
        ))
    elif args.splits_directory:
        docs_db = asyncio.run(embedding_service.create_embedding_database_from_splits(
            splits_directory=args.splits_directory,
            model_name=args.model_name,
            chunk_size=BATCH_SIZE,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory
        ))
    else:
        if args.file_types is None:
            split_docs = documents_store_service.load_supported_documents(dir_path=args.dir_path)
        else:
            file_loader_query = FileLoaderQuery.get_file_loader_query(file_types=args.file_types, file_patterns=args.file_patterns, logging=logging)
            split_docs = documents_store_service.load_documents(dir_path=args.dir_path, file_loader_query=file_loader_query)

        docs_db = asyncio.run(embedding_service.create_embedding_database(
            documents=split_docs,
            model_name=args.model_name,
            chunk_size=BATCH_SIZE,
            collection_name=args.collection_name,
            persist_directory=args.persist_directory
        ))

    embedding_service.create_manifest(collection_name=args.collection_name, model_name=args.model_name, persist_directory=args.persist_directory)

    elapsed_time_msg = get_elapse_time_message(start_time=start_time)

    if docs_db is None:
        logging.info("The vectorstore is empty.")
    else:
        logging.info(f"Finished the creation of vectorstore creation in {elapsed_time_msg}.")

        # Test a new vectorstore
        doc_ids = docs_db.get()["ids"]
        logging.info(f"The vectorstore stores {len(doc_ids)} documents")

        if args.test_question is not None:
            retriever = docs_db.as_retriever()
            answer = retriever.invoke(args.test_question)
            logging.info(f"ANSWER: {answer[0].page_content}")

        logging.info(f"The vectorstore: {docs_db}")