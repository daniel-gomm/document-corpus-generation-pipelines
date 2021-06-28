import abc
import os
import logging
from .DocumentFields import MetadataFields
from pathlib import Path
from os import scandir
from os.path import isfile, join, exists
import traceback
from typing import List, Dict
from .document_store import ElasticsearchDocumentStore
import bs4
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

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
    
    def __init__(self, folderpath:str, line_mode:bool=False):
        """Adapter for unarXive textfiles.

        Args:
            folderpath (str): Directory of the unarXive fulltexts.
        """        
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._no_unprocessed_files = len(list(filter(lambda x: x.endswith(".txt"), files)))
        self._file_iterator = get_files(Path(folderpath))
        self._line_mode = line_mode
    
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
                        if self._line_mode:
                            document["text"] = "".join(text)
                        else:
                            document["text"] = "".join(text).replace("\n", " ")
                    document["meta"][MetadataFields.ARXIVE_ID.value] = path.split("/")[-1][:-4]
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
    
    def __init__(self, folderpath:str, split_len:int=100):
        """**Not Implemented** Adapter for GROBID output (TEI XML Files).

        Args:
            folderpath (str): Directory holding the textfiles.
        """   
        self._folderpath = folderpath
        files = os.listdir(folderpath)
        self._no_unprocessed_files = len(list(filter(lambda x: x.endswith(".xml"), files)))
        self._file_iterator = get_files(Path(folderpath))
        self._split_len = split_len
    
    def reset(self):
        files = os.listdir(self._folderpath)
        self._no_unprocessed_files = len(list(filter(lambda x: x.endswith(".xml"), files)))
        self._file_iterator = get_files(Path(self._folderpath))
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        documents = []
        counter = 0
        doc = next(self._file_iterator, None)
        while counter < no_documents and doc:
            path = str(doc)
            if path.endswith(".xml"):
                try:
                    counter += 1
                    document = {}
                    document["meta"] = {}
                    with open(doc, "r") as paper:
                        document["text"] = "\n".join(self._extract_paragraphs(BeautifulSoup(paper, 'lxml')))
                    document["meta"][MetadataFields.MAKG_ID.value] = path.split("/")[-1][:-8]
                    documents.append(document)
                    self._no_unprocessed_files -= 1
                except:
                    logging.error(traceback.format_exc())
                if counter < no_documents:
                    doc = next(self._file_iterator, None)
        return documents
    
    def generate_and_split_documents(self, no_documents: int) -> List[Dict]:
        documents = []
        counter = 0
        doc = next(self._file_iterator, None)
        while counter < no_documents and doc:
            path = str(doc)
            if path.endswith(".xml"):
                try:
                    counter += 1
                    with open(doc, "r") as paper:
                        paragraphs = self._extract_paragraphs(BeautifulSoup(paper, 'lxml'))
                    split_id = 0
                    p_cleaned = []
                    for paragraph in paragraphs:
                        p_tokenized = word_tokenize(paragraph)
                        if len(p_tokenized) < 5:
                            continue
                        elif len(p_tokenized) < self._split_len:
                            p_cleaned.append(paragraph)
                        else:
                            p_sentences = sent_tokenize(paragraph)
                            current_sent = ""
                            len_current = 0
                            for sent in p_sentences:
                                words = word_tokenize(sent)
                                len_sent = len(words)
                                if len_current >= self._split_len:
                                    while len_sent >= self._split_len:
                                        w = words[0:np.minimum(self._split_len, len(words))]
                                        p_cleaned.append(TreebankWordDetokenizer().detokenize(w))
                                        words = words[self._split_len:]
                                        len_sent = len(words)
                                elif len_current + len_sent < self._split_len:
                                    current_sent += " " + sent
                                    len_current += len_sent
                                else:
                                    p_cleaned.append(current_sent)
                                    current_sent = sent
                                    len_current = len_sent
                    for paragraph in p_cleaned:
                        document = {}
                        document["meta"] = {}
                        document["meta"][MetadataFields.MAKG_ID.value] = path.split("/")[-1][:-8]
                        document["meta"]["_split_id"] = split_id
                        split_id += 1
                        document["text"] = paragraph
                        documents.append(document)
                    self._no_unprocessed_files -= 1
                except:
                    logging.error(traceback.format_exc())
                if counter < no_documents:
                    doc = next(self._file_iterator, None)
        return documents
    
    def _extract_text(self, soup:BeautifulSoup):
        divs_text = []
        for div in soup.body.find_all("div"):
            if not div.get("type"):
                for child in div.children:
                    if(not child.name == "listbibl"):
                        if(isinstance(child, bs4.element.NavigableString)):
                            divs_text.append(str(child))
                        elif(isinstance(child, bs4.element.Tag)):
                            divs_text.append(child.text)
        return " ".join(divs_text)
    
    def _extract_paragraphs(self, soup:BeautifulSoup)->List:
        divs_text = []
        for div in soup.body.find_all("div"):
            if not div.get("type"):
                for child in div.children:
                    if(not child.name == "listbibl"):
                        if(isinstance(child, bs4.element.NavigableString)):
                            divs_text.append(str(child))
                        elif(isinstance(child, bs4.element.Tag)):
                            divs_text.append(child.text)
        return divs_text
    
    def __len__(self) -> int:
        return self._no_unprocessed_files

class ElasticsearchAdapter(Adapter):

    def __init__(self, document_store:ElasticsearchDocumentStore, filters:Dict[str, List[str]]=None, batch_size: int = 10000):
        """Adapter for the Haystack Elasticsearch DocumentStore.

        Args:
            document_store (ElasticsearchDocumentStore): Elasticsearch DocumentStore from which documents should be processed.
            filters(Dict[str, List[str]]): Optional filters to narrow down the documents to return. Example: {"name": ["some", "more"], "category": ["only_one"]}git ch
            batch_size (int, optional): Batch size in which documents should be retrieved from the document store. Defaults to 10000.
        """        
        self._document_store = document_store
        self._batch_size = batch_size
        self._filters = filters
        self._generator = self._document_store.get_all_documents_generator(batch_size = batch_size, filters=filters)
        self._unprocessed_documents = self._document_store.get_document_count(filters=filters)
    
    def reset(self):
        self._generator = self._document_store.get_all_documents_generator(batch_size = self._batch_size, filters=self._filters)
        self._unprocessed_documents = self._document_store.get_document_count(filters=self._filters)
    
    def generate_documents(self, no_documents: int) -> List[Dict]:
        docs = []
        while len(docs) < no_documents:
            new_document = next(self._generator, None)
            if new_document is None:
                break
            docs.append(new_document.to_dict())
        self._unprocessed_documents -= len(docs)
        return docs
    
    def __len__(self):
        return self._unprocessed_documents