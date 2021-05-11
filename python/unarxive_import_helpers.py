from arxive_metadata import rocksDB
from arXive_categories import cs_categories
import json
import re
import os
from alive_progress import alive_bar
from timeit import default_timer as timer
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor import PreProcessor

def delete_citations(string:str):
    return re.sub(r"\{{(.*?)\}}", "", string)

class Document:
    def __init__(self, filepath:str):
        self._filepath = filepath
        self._text = self._extract_text()
        self._id = self._extract_id()
        self._meta = {}
        
    def set_meta(self, metadata:(str or dict)):
        if(isinstance(metadata, str)):
            self._meta = json.loads(metadata)
        if(isinstance(metadata, dict)):
            self._meta = metadata
    
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

def contains_cs_category(text:str):
    return any(substring in text for substring in cs_categories)

def put_into_documentstore(document_store, documents:list):
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0
    )
    docs_to_write = []
    for d in documents:
        docs_to_write.extend(processor.process(d.get_document_as_dict()))
    document_store.write_documents(docs_to_write)

def process_docements(directory:str, files:list, db:rocksDB.RocksDBAdapter, batch_size:int=100):
    statistics = {
        "no_original_papers": files.__len__(),
        "no_output_papers": 0,
        "no_paper_without_metadata":0,
        "elapsed_time":0
    }
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
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
            #Write documents into ElasticSearch DocumentStore
            put_into_documentstore(document_store, cs_documents)
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