import abc
import re
import json
import torch
import nltk

from typing import List, Dict
from haystack.preprocessor import PreProcessor
from arxive_metadata.rocksDB import RocksDBAdapter
from transformers import pipeline
from transformers import BertForSequenceClassification
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
#from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

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
        for document in documents:
            received_meta = metadata[document["meta"][self._id_field]]
            if(received_meta == "Data unavailable"):
                document["meta"].update({"unavailable metadata":"Metadata not found in database."})
                if self._discard_missing:
                    documents.pop(document)
            elif received_meta != "Data unavailable":
                document["meta"].update(json.loads(received_meta))
        return documents
    

#Text Processors

class TextKeywordCut(Processor):

    def __init__(self, keyword:str, cut_off_upper_part:bool = True):
        self._keyword = keyword
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
                    text = text[:(len(text_substring)+len(self._keyword))]
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
    def __init__(self):
        #self._keyword = keyword
        #self._cut__off_upper_part = cut_off_upper_part
        myClass=Classification()

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            sentences=sent_tokenize(document["text"])
            #tokens = word_tokenize(document["text"])
            labels=myClass.classify(sentences) #instanz
            a=0
            b=0+len(str.split(sentences[0]))
            textIMRaD=f"{a} {b} {labels[0]} "
            for l in range (1,len(labels)):
                a=b
                b=b+len(str.split(sentences[l]))
                textIMRaD+=f"{a} {b} {labels[l]} "
            document["meta"]["IMRAD"]=textIMRaD
        return documents


class Classification:
    def __init__(self):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4, output_attentions=False, output_hidden_states=False)
        model.load_state_dict(torch.load("/home/kd-sem/GroupD/finetuned_BERT_epoch_5.model", map_location=torch.device('cpu')))
        global classificationPipeline
        classificationPipeline=pipeline("text-classification", model=model, tokenizer='bert-base-uncased')
    

    def classify(self,string):
        labels=[]
        a=classificationPipeline(string)
        for i in range(0,len(a)):
            b=a[i]['label']
            if b=="LABEL_0":
                labels.append("intro")
            if b=="LABEL_1":
                labels.append("methods")
            if b=="LABEL_2":
                labels.append("results")
            if b=="LABEL_3":
                labels.append("discussion")
        return labels