import abc
import os
from typing import List, Dict
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

class Adapter(metaclass=abc.ABCMeta):
    #Adapter to input data
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'generate_documents') and callable(subclass.generate_documents) and hasattr(subclass, '__lon__') and callable(subclass.__lon__) and hasattr(subclass, 'reset') and callable(subclass.reset) or NotImplemented)

    @abc.abstractmethod
    def generate_documents(self, no_documents:int) -> List[Dict]:
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class UnarxiveAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._unprocessed_files = list(filter(lambda x: x.endswith(".txt"), files))
    
    def reset(self):
        files = os.listdir(self._folderpath)
        self._unprocessed_files = list(filter(lambda x: x.endswith(".txt"), files))

    def generate_documents(self, no_documents: int) -> List[Dict]:
        documents = []
        counter = 0
        if not os.path.exists(self._folderpath):
            os.makedirs(self._folderpath)
        while counter < no_documents and self._unprocessed_files:
            counter += 1
            document = {}
            document["meta"] = {}
            file = self._unprocessed_files.pop()
            filename = "{}/{}".format(self._folderpath, os.fsdecode(file))
            with open(filename, "r") as paper:
                text = paper.readlines()
                document["text"] = "".join(text).replace("\n", " ")
            document["meta"]["arixive-id"] = filename.split("/")[-1][:-4]
            documents.append(document)
        return documents
    
    def __len__(self) -> int:
        return len(self._unprocessed_files)


class TextfileAdapter(Adapter):
    
    def __init__(self, folderpath:str):
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


class UnpaywallAdapter(Adapter):
    
    def __init__(self, folderpath:str):
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