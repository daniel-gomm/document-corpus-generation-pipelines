import abc
from typing import List
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import torch


class ClassificationHandler(metaclass=abc.ABCMeta):
    #Interface for IMRaD Classification-Wrapper
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'classify') and callable(subclass.classify) or NotImplemented)

    @abc.abstractmethod
    def classify(self, text_to_classify:str) -> List:
        raise NotImplementedError


class BERTClassificationHandler(ClassificationHandler):
    def __init__(self, bert_filename:str):
        self._model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4, output_attentions=False, output_hidden_states=False)
        #self._model.load_state_dict(torch.load(bert_filename, map_location=torch.device('cpu')))
        self._model.load_state_dict(torch.load(bert_filename, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        self._classificationPipeline = TextClassificationPipeline(model=self._model, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'))

    def classify(self,text_to_classify:str) -> List:
        output_labels=[]
        labels = self._classificationPipeline(text_to_classify)
        for entry in labels:
            label = entry['label']
            if label == "LABEL_0":
                output_labels.append("intro")
            if label == "LABEL_1":
                output_labels.append("methods")
            if label == "LABEL_2":
                output_labels.append("results")
            if label == "LABEL_3":
                output_labels.append("discussion")
        return output_labels