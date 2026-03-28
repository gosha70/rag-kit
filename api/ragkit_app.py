# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the CC-BY-SA-4.0 license.
import time
import argparse
import logging

from unstructured.document_splitter import DocumentSplitter
from vectorstores.documents_store_service import DocumentsStoreService
from unstructured.file_loader_query import FileLoaderQuery
from rag.embeddings.chroma.embeddings_chroma_constants import get_elapse_time_message

def main():        
    """
    Main function to load documents, split them, and save the splits to disk.
    """
    # Set the logging level to INFO    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create the parser
    parser = argparse.ArgumentParser(description="Creating the vectorsstore ...")

    # Add the arguments
    parser.add_argument(
        '--dir_path', 
        type=str, 
        help='The root directory where to look for documents.', 
        default="."
    )
    parser.add_argument(
        '--file_types', 
        type=str, nargs='+', 
        help='The list of file extensions without the dot: md; java; xml; html; pdf'
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
        help='(Optional) The path to the directory where unstructured document splits are saved.', 
        default=None
    )
  
    # Parse the arguments
    args = parser.parse_args()

    logging.info(f"Searching and processing documents with the arguments: {args}")
    documents_store_service = DocumentsStoreService(document_splitter=DocumentSplitter(logging), logging=logging)
    # Load and split documents
    start_time = time.time()
    if args.file_types is None:
        split_docs = documents_store_service.load_supported_documents(args.dir_path)
    else:
        file_loader_query = FileLoaderQuery.get_file_loader_query(args.file_types, args.file_patterns, logging=logging)    
        split_docs = documents_store_service.load_documents(dir_path=args.dir_path, file_loader_query=file_loader_query)

    elapsed_time_msg = get_elapse_time_message(start_time=start_time)
    logging.info(f"Finished the document loading in {elapsed_time_msg}.")

    if split_docs is not None:
        # Save split documents to disk
        doc_dir = documents_store_service.save_splits_to_disk(split_docs, args.persist_directory)
        logging.info(f"{len(split_docs)} documet chunks are saved in {doc_dir}.")
    else:
        logging.warn("No documents were loaded or split.")

if __name__ == "__main__":
    main() 