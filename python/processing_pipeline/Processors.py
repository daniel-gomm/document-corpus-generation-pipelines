import abc
import re
import json
import nltk
import logging
logging.basicConfig(filename="processor_logs.log", level=logging.DEBUG)

from pandas import DataFrame
from typing import List, Dict
from haystack.preprocessor import PreProcessor
from arxive_metadata.rocksDB import RocksDBAdapter

from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from nltk.tokenize.punkt import PunktSentenceTokenizer
from imrad_classification import ClassificationHandler, BERTClassificationHandler

class Processor(metaclass=abc.ABCMeta):
    #Interface for processing steps
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process') and callable(subclass.process) or NotImplemented)
    
    @abc.abstractmethod
    def process(self, documents:List[Dict]) -> List[Dict]:
        raise NotImplementedError

#General Processors

class HaystackPreProcessor(Processor):

    def __init__(self, preprocessor:PreProcessor):
        self._preprocessor = preprocessor
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        docs = []
        for document in documents:
            docs.extend(self._preprocessor.process(document))
        return docs


#Metadata Processors

class MetadataFieldDiscarder(Processor):

    def __init__(self, fields_to_discard:List):
        self._fields_to_discard = fields_to_discard

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            for field in self._fields_to_discard:
                document["meta"].pop(field, None)
        return documents


class MetadataArxiveEnricher(Processor):

    def __init__(self, id_field:str, db:RocksDBAdapter, discard_entries_without_metadata:bool=True):
        self._db = db
        self._id_field = id_field
        self._discard_missing = discard_entries_without_metadata
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        ids = list(map(lambda doc: doc["meta"][self._id_field], documents))
        response = self._db.get_all(ids)
        metadata = json.loads(response.text)
        docs_to_return = []
        for document in documents:
            received_meta = metadata[document["meta"][self._id_field]]
            if(received_meta == "Data unavailable"):
                document["meta"].update({"unavailable metadata":"Metadata not found in database."})
                if not self._discard_missing:
                    docs_to_return.append(document)
            elif received_meta != "Data unavailable":
                document["meta"].update(json.loads(received_meta))
                docs_to_return.append(document)
        return docs_to_return
    
class MetadataMagArxiveLinker(Processor):

    def __init__(self, dataframe:DataFrame, column_to_match:str, column_to_add:str, field_to_match:str = "arixive-id"):
        self._column_to_match = column_to_match
        self._column_to_add = column_to_add
        self._dataframe = dataframe[[self._column_to_add, self._column_to_match]]
        self._field_to_match = field_to_match
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        docs_to_return = []
        for document in documents:
            try:
                document["meta"][self._column_to_add] = self._find_match(document["meta"][self._field_to_match])
                docs_to_return.append(document)
            except IndexError:
                logging.info("No matching id found for {}".format(document["meta"][self._field_to_match]))
        return documents
    
    def _find_match(self, value):
        matching_value = self._dataframe[self._dataframe[self._column_to_match] == value][self._column_to_add].iloc[0]
        return int(matching_value)

#Text Processors

class TextKeywordCut(Processor):

    def __init__(self, keyword:str, cut_off_upper_part:bool = True):
        self._keyword = keyword.lower()
        self._cut__off_upper_part = cut_off_upper_part

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            text = document["text"]
            if self._keyword in text.lower():
                if self._cut__off_upper_part:
                    text_substring = text.lower().partition(self._keyword)[2]
                    text = text[-(len(text_substring)+len(self._keyword)):]
                else:
                    text_substring = text.lower().partition(self._keyword)[0]
                    text = text[0:(len(text_substring))]
                document["text"] = text
        return documents
    


class TextReplaceFilter(Processor):

    def __init__(self, filter:str, replacement:str):
        self._filter = filter
        self._replacement = replacement
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            document["text"] = re.sub(self._filter, self._replacement, document["text"])
        return documents


class TextAppendMetadataField(Processor):

    def __init__(self, field_to_attach:str, metdata_field_content_before_text:bool = True):
        self._field_to_attach = field_to_attach
        self._metdata_field_content_before_text = metdata_field_content_before_text
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            meta_field_content = document["meta"][self._field_to_attach]
            if meta_field_content != None:
                if self._metdata_field_content_before_text:
                    document["text"] = meta_field_content + " " + document["text"]
                else:
                    document["text"] += " " + meta_field_content
        return documents

#Filter Processors

class FilterOnMetadataValue(Processor):

    def __init__(self, metadata_field:str, values:List, discard_docs_wo_value:bool=True):
        self._metadata_field = metadata_field
        self._values = values
        self._discard_wo_values = discard_docs_wo_value
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        if self._discard_wo_values:
            return list(filter(lambda d: self._contains_value(d["meta"][self._metadata_field]), documents))
        else:
            return list(filter(lambda d: not self._contains_value(d["meta"][self._metadata_field]), documents))
    
    def _contains_value(self, text:str):
        return any(substring in text for substring in self._values)

class IMRaDClassification(Processor):
    def __init__(self, classification_handler:ClassificationHandler = BERTClassificationHandler("/home/daniel/BERT_copy.model")):
        #self._keyword = keyword
        #self._cut__off_upper_part = cut_off_upper_part
        self._classification_handler=classification_handler

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            sentences = sent_tokenize(document["text"])
            #tokens = word_tokenize(document["text"])
            labels = self._classification_handler.classify(sentences) #instance
            first_token_in_sentence = 0
            last_token_in_sentence = 0
            classification_result = []
            for index, label in enumerate(labels):
                last_token_in_sentence = last_token_in_sentence+len(str.split(sentences[index]))
                classification_result.append(
                    {"first_token": first_token_in_sentence,
                    "last_token": last_token_in_sentence,
                    "label": label})
                first_token_in_sentence = last_token_in_sentence
            document["meta"]["IMRAD"] = classification_result
        return documents