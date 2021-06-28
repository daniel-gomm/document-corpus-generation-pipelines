from arxive_metadata import rocksDB
from arXive_categories import cs_categories
import json
import re
import os
import abc
from alive_progress import alive_bar
from timeit import default_timer as timer
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

def delete_citations(string:str):
    return re.sub(r"\{{(.*?)\}}", "", string)

def clean_metadata(meta:dict):
    
    for key in ["update_date", "versions", "comments", "report-no", "license"]:
        try:
            meta.pop(key)
        except:
            print("Could not clean up metadata.")
    return meta

class Document:
    def __init__(self, filepath:str):
        self._filepath = filepath
        self._text = self._extract_text()
        self._id = self._extract_id()
        self._meta = {}
        
    def set_meta(self, metadata:(str or dict)):
        if(isinstance(metadata, str)):
            self._meta = clean_metadata(json.loads(metadata))
        if(isinstance(metadata, dict)):
            self._meta = clean_metadata(metadata)
    
    def get_document_as_dict(self):
        return {
            "text": self._text,
            "meta": self._meta
        }
    
    def _extract_text(self):
        with open(self._filepath, "rt") as file:
            text = file.readlines()
        #Delete citations from the text
        text = delete_citations("".join(text).replace("\n", " "))
        if "introduction" in text.lower():
            text_substring = text.lower().partition("introduction")[2]
            text = text[-(text_substring.__len__()+12):]
        return text
    
    def _extract_id(self):
        return self._filepath.split("/")[-1][:-4]

class document_saver_interface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'store_data') and callable(subclass.store_data) or NotImplemented)

    @abc.abstractmethod
    def store_data(self, documents:list):
        raise NotImplementedError

class document_store_document_saver(document_saver_interface):

    def __init__(self, document_store, processor):
        self._document_store = document_store
        self._processor = processor
    
    def store_data(self, documents:list):
        docs_to_write = []
        for d in documents:
            docs_to_write.extend(self._processor.process(d.get_document_as_dict()))
        self._document_store.write_documents(docs_to_write)

class textfile_document_saver(document_saver_interface):

    def __init__(self, folderpath:str, processor):
        self._processor = processor
        self._folderpath = folderpath
    
    def store_data(self, documents:list):
        for document in documents:
            docs_to_write = self._processor.process(document.get_document_as_dict())
            for index, doc_part in enumerate(docs_to_write):
                with open("{}/{}.{}.txt".format(self._folderpath, document._id, index), "w") as file:
                    file.writelines(json.dumps(doc_part))

def contains_cs_category(text:str):
    return any(substring in text for substring in cs_categories)

def process_documents(directory:str, files:list, db:rocksDB.RocksDBAdapter, document_saver:document_saver_interface, batch_size:int=100):
    statistics = {
        "no_original_papers": files.__len__(),
        "no_output_papers": 0,
        "no_paper_without_metadata":0,
        "elapsed_time":0
    }
    
    time_start = timer()
    print("Processing {} documents in batches of {}.".format(statistics["no_original_papers"], batch_size))
    dummy = []
    with alive_bar(int(statistics["no_original_papers"]/batch_size)) as bar:
        while files:
            counter = 0
            documents = []
            while counter < batch_size and files:
                counter += 1
                file = files.pop()
                filename = directory + "/" + os.fsdecode(file)
                d = Document(filename)
                documents.append(d)
            #Get metadata from the database
            add_metadata_to_documents(documents, db)
            #Filter to only include publications for which metadata is available
            docs_with_metadata = list(filter(lambda d: "unavailable metadata" not in d._meta, documents))
            statistics["no_paper_without_metadata"] += (batch_size - docs_with_metadata.__len__())
            #Filter to only include CS documents
            cs_documents = list(filter(lambda d: contains_cs_category(d._meta["categories"]), docs_with_metadata))
            statistics["no_output_papers"] += cs_documents.__len__()
            bar()
            #Write out documents
            document_saver.store_data(cs_documents)
    statistics["elapsed_time"] = timer() - time_start
    return statistics

def add_metadata_to_documents(documents:list, db:rocksDB.RocksDBAdapter):
    ids = list(map(lambda doc: doc._id, documents))
    response = db.get_all(ids)
    metadata = json.loads(response.text)
    for document in documents:
        received_meta = metadata[document._id]
        if(received_meta == "Data unavailable"):
            document.set_meta({"unavailable metadata":"Metadata not found in database."})
        else:
            document.set_meta(received_meta)