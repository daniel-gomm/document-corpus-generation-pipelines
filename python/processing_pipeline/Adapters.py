import abc
import os
import logging
from pathlib import Path
from os import scandir
from os.path import isfile, join, exists
import traceback
from typing import List, Dict
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

#Helpers

def get_files(path:Path):
    if exists(path):
        for file in scandir(path):
            full_path = join(path, file.name)
            if isfile(full_path):
                yield full_path
    else:
        print('Path doesn\'t exist')

class Adapter(metaclass=abc.ABCMeta):
    #Adapter to input data
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'generate_documents') and callable(subclass.generate_documents) and hasattr(subclass, '__lon__') and callable(subclass.__lon__) and hasattr(subclass, 'reset') and callable(subclass.reset) or NotImplemented)

    @abc.abstractmethod
    def generate_documents(self, no_documents:int) -> List[Dict]:
        """Generates a given number of documents.

        Args:
            no_documents (int): Number of documents that should be generated.

        Raises:
            NotImplementedError: Raised if subclass doesn't implement method.

        Returns:
            List[Dict]: Documents represented by a dictionary.
        """        
        raise NotImplementedError
    
    def reset(self):
        """Resets the adapter to its initial state.

        Raises:
            NotImplementedError: Raised if subclass doesn't implement method.
        """        
        raise NotImplementedError
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class UnarxiveAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        """Adapter for unarXive textfiles.

        Args:
            folderpath (str): Directory of the unarXive fulltexts.
        """        
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._no_unprocessed_files = len(list(filter(lambda x: x.endswith(".txt"), files)))
        self._file_iterator = get_files(Path(folderpath))
    
    def reset(self):
        files = os.listdir(self._folderpath)
        self._no_unprocessed_files = len(list(filter(lambda x: x.endswith(".txt"), files)))
        self._file_iterator = get_files(Path(self._folderpath))

    def generate_documents(self, no_documents: int) -> List[Dict]:
        documents = []
        counter = 0
        doc = next(self._file_iterator, None)
        while counter < no_documents and doc:
            path = str(doc)
            if path.endswith(".txt"):
                try:
                    counter += 1
                    document = {}
                    document["meta"] = {}
                    with open(doc, "r") as paper:
                        text = paper.readlines()
                        document["text"] = "".join(text).replace("\n", " ")
                    document["meta"]["arixive-id"] = path.split("/")[-1][:-4]
                    documents.append(document)
                    self._no_unprocessed_files -= 1
                except:
                    logging.error(traceback.format_exc())
                if counter < no_documents:
                    doc = next(self._file_iterator, None)
        return documents
    
    def __len__(self) -> int:
        return self._no_unprocessed_files


class TextfileAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        """**Not Implemented** Adapter for textfiles that contain a json representation of the document.

        Args:
            folderpath (str): Directory holding the textfiles.
        """        
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._unprocessed_files = list(filter(lambda x: x.endswith(".txt"), files))
    
    def reset(self):
        files = os.listdir(self._folderpath)
        self._unprocessed_files = list(filter(lambda x: x.endswith(".txt"), files))
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        #TODO: Implement
        return super().generate_documents(no_documents)
    
    def __len__(self) -> int:
        return len(self._unprocessed_files)


class GrobidAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        """**Not Implemented** Adapter for GROBID output (TEI XML Files).

        Args:
            folderpath (str): Directory holding the textfiles.
        """   
        self._folderpath = folderpath
    
    def reset(self):
        #TODO: Implement
        return super().reset()
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        #TODO: Implement
        return super().generate_documents(no_documents)
    
    def __len__(self) -> int:
        #TODO: Implement
        return super().__len__()

class ElasticsearchAdapter(Adapter):

    def __init__(self, document_store:ElasticsearchDocumentStore, batch_size: int = 10000):
        """Adapter for the Haystack Elasticsearch DocumentStore.

        Args:
            document_store (ElasticsearchDocumentStore): Elasticsearch DocumentStore from which documents should be processed.
            batch_size (int, optional): Batch size in which documents should be retrieved from the document store. Defaults to 10000.
        """        
        self._document_store = document_store
        self._batch_size = batch_size
        self._generator = self._document_store.get_all_documents_generator(batch_size = batch_size)
        self._unprocessed_documents = self._document_store.get_document_count()
    
    def reset(self):
        self._generator = self._document_store.get_all_documents_generator(batch_size = self._batch_size)
        self._unprocessed_documents = self._document_store.get_document_count()
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        docs = []
        while len(docs) < no_documents:
            new_document = next(self._generator, None)
            if new_document is None:
                break
            docs.append(new_document)
        self._unprocessed_documents -= len(docs)
        return docs
    
    def __len__(self):
        return self._unprocessed_documents