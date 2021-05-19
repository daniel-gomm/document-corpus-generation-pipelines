import abc
import re
import json
from typing import List, Dict
from haystack.preprocessor import PreProcessor
from .arxive_metadata.rocksDB import RocksDBAdapter

class Processor(metaclass=abc.ABCMeta):
    #Interface for processing steps
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process') and callable(subclass.process) or NotImplemented)

    @abc.abstractmethod
    def process(self, document:Dict) -> Dict:
        raise NotImplementedError
    
    @abc.abstractmethod
    def process(self, documents:List[Dict]) -> List[Dict]:
        raise NotImplementedError

#General Processors

class HaystackPreProcessor(Processor):

    def __init__(self, preprocessor:PreProcessor):
        self._preprocessor = preprocessor
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        return self._preprocessor.process(documents)
    
    def process(self, document:Dict) -> Dict:
        return self._preprocessor.process(document)

#Metadata Processors

class MetadataFieldDiscarder(Processor):

    def __init__(self, fields_to_discard:List):
        self._fields_to_discard = fields_to_discard

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            for field in self._fields_to_discard:
                document["meta"].pop(field, None)
        return documents
    
    def process(self, document:Dict) -> Dict:
        return self.process([document]).pop()


class MetadataArxiveEnricher(Processor):

    def __init__(self, id_field:str, db:RocksDBAdapter, discard_entries_without_metadata:bool=True):
        self._db = db
        self._id_field = id_field
        self._discard_missing = discard_entries_without_metadata
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        ids = list(map(lambda doc: doc["meta"][self._id_field], documents))
        response = self._db.get_all(ids)
        metadata = json.loads(response.text)
        for document in documents:
            received_meta = metadata[document["meta"][self._id_field]]
            if(not self._discard_missing and received_meta == "Data unavailable"):
                document["meta"].update({"unavailable metadata":"Metadata not found in database."})
            elif received_meta != "Data unavailable":
                document["meta"].update(received_meta)
        return documents
    
    def process(self, document:Dict) -> Dict:
        return self.process([document]).pop()

#Text Processors

class TextKeywordCut(Processor):

    def __init__(self, keyword:str, cut__off_upper_part:bool = True):
        self._keyword = keyword
        self._cut__off_upper_part = cut__off_upper_part

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            text = document["text"]
            if self._keyword in text.lower():
                if self._cut__off_upper_part:
                    text_substring = text.lower().partition(self._keyword)[2]
                    text = text[-(len(text_substring)+len(self._keyword)):]
                else:
                    text_substring = text.lower().partition(self._keyword)[0]
                    text = text[:(len(text_substring)+len(self._keyword))]
                document["text"] = text
        return documents
    
    def process(self, document:Dict) -> Dict:
        return self.process([document]).pop()


class TextReplaceFilter(Processor):

    def __init__(self, filter:str, replacement:str):
        self._filter = filter
        self._replacement = replacement
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            document["text"] = re.sub(self._filter, self._replacement, document["text"])
        return documents
    
    def process(self, document:Dict) -> Dict:
        return self.process([document]).pop()

#Filter Processors

class FilterOnMetadataValue(Processor):

    def __init__(self, metadata_field:str, values:List, discard_docs_wo_value:bool=True):
        self._metadata_field = metadata_field
        self._values = values
        self._discart_wo_values = discard_docs_wo_value
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            if self._discart_wo_values and any(item in document["meta"][self._metadata_field] for item in self._values):
                documents.remove(document)
            elif not self._discart_wo_values and not any(item in document["meta"][self._metadata_field] for item in self._values):
                documents.remove(document)
        return documents
    
    def process(self, document:Dict) -> Dict:
        return self.process([document]).pop()