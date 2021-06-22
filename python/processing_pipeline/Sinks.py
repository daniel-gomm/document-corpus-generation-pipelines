import abc
import json
from typing import List, Dict
from haystack_copy import ElasticsearchDocumentStore

class Sink(metaclass=abc.ABCMeta):
    #Interface for sinks (saving data)
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process') and callable(subclass.process) or NotImplemented)
    
    @abc.abstractmethod
    def process(self, documents:List[Dict]):
        """Processes documents through the sink.

        Args:
            documents (List[Dict]): Documents to be processed.

        Raises:
            NotImplementedError: Raised if sublclass doesn't implement the method. 
        """        
        raise NotImplementedError


class ElasticsearchSink(Sink):

    def __init__(self, document_store:ElasticsearchDocumentStore, batch_size:int=10000):
        """Wrapper for the Haystack Elasticsearch Document Store.

        Args:
            document_store (ElasticsearchDocumentStore): Preconfigured Haystack Elasticsearch Documentstore.
            batch_size (int): Number of documents that are passed to Elasticsearch's bulk function at a time.
        """        
        self._document_store = document_store
        self._batch_size = batch_size
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        self._document_store.write_documents(documents, batch_size=self._batch_size)


class TextfileSink(Sink):

    def __init__(self, folderpath:str, filename_meta_fields:List[str]):
        """Saves output to textfiles.

        Args:
            folderpath (str): Directory to which files should be saved to.
            filename_meta_fields (List[str]): Metadata fields specifiying the filenames.
        """        
        self._folderpath = folderpath
        self._filename_meta_fields = filename_meta_fields
    
    def process(self, documents: List[Dict]):
        for document in documents:
            filename = "{}/".format(self._folderpath)
            for meta_field in self._filename_meta_fields:
                filename += "{}.".format(document["meta"][meta_field])
            filename += "json"
            with open(filename, "w") as file:
                    file.writelines(json.dumps(document))


class CSVMetadataSink(Sink):

    def __init__(self, filepath:str, metadata_field:str) -> None:
        """Saves a metadata field into a CSV file.

        Args:
            filepath (str): Path under which the CSV-file should be stored.
            metadata_field (str): Metadatafield to be saved.
        """        
        self._file = open(filepath, "a+")
        self._metadata_field = metadata_field
    
    def process(self, documents: List[Dict]):
        batch_set = set()
        for document in documents:
            try:
                batch_set.add(document["meta"][self._metadata_field])
            except:
                pass
        for entry in batch_set:
            self._file.write(str(entry) + ",\n")
    
    def __del__(self):
        self._file.close()

class PrintSink(Sink):

    def process(self, documents: List[Dict]):
        print(documents)
