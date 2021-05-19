import abc
import os
from typing import List, Dict

class Adapter(metaclass=abc.ABCMeta):
    #Adapter to input data
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'generate_documents') and callable(subclass.generate_documents) and hasattr(subclass, '__lon__') and callable(subclass.__lon__) or NotImplemented)

    @abc.abstractmethod
    def generate_documents(self, no_documents:int) -> List[Dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class UnarxiveAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._unprocessed_files = list(filter(lambda x: x.endswith(".txt"), files))

    def generate_documents(self, no_documents: int) -> List[Dict]:
        documents = []
        counter = 0
        while counter < no_documents and self._unprocessed_files:
            counter += 1
            document = {}
            document["meta"] = {}
            file = self._unprocessed_files.pop()
            filename = "{}/{}".format(self._folderpath, os.fsdecode(file))
            with open(filename, "r") as paper:
                document["text"] = paper.readlines()
            document["meta"]["id"] = filename.split("/")[-1][:-4]
            documents.append(document)
        return documents
    
    def __len__(self) -> int:
        return self._unprocessed_files.__len__()


class UnpaywallAdapter(Adapter):
    
    def __init__(self, folderpath:str):
        self._folderpath = folderpath
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        #TODO: Implement
        return super().generate_documents(no_documents)
    
    def __len__(self) -> int:
        #TODO: Implement
        return super().__len__()