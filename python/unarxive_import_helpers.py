from arxive_metadata import rocksDB
import json
import re
import os

def delete_citations(string:str):
    return re.sub(r"\{{(.*?)\}}", "", string)

class Document:
    def __init__(self, filepath:str):
        self._filepath = filepath
        self._text = self._extract_text()
        self._id = self._extract_id()
        self._meta = ""
        
    def set_meta(self, metadata:(str, dict)):
        if(isinstance(metadata, str)):
            self._meta = metadata
        if(isinstance(metadata, dict)):
            self._meta = json.dumps(metadata)
    
    def get_document_as_dict(self):
        return {
            "text": self._text,
            "meta": self._meta
        }
    
    def _extract_text(self):
        with open(self._filepath, "rt") as file:
            text = file.readlines()
        return delete_citations("".join(text).replace("\n", " "))
    
    def _extract_id(self):
        return self._filepath.split("/")[-1][:-4]

def process_docements(directory:str, files:list, db:rocksDB.RocksDBAdapter, batch_size:int=100):
    print("Processing {} documents in batches of {}.".format(files.__len__(), batch_size))
    dummy = []
    while files:
        counter = 0
        documents = []
        while counter < batch_size and files:
            file = files.pop()
            filename = directory + "/" + os.fsdecode(file)
            d = Document(filename)
            documents.append(d)
        add_metadata_to_documents(documents, db)
        dummy = dummy + documents
    return dummy

def add_metadata_to_documents(documents:list, db:rocksDB.RocksDBAdapter):
    ids = list(map(lambda doc: doc._id, documents))
    response = db.get_all(ids)
    metadata = json.loads(response.text)
    for document in documents:
        received_meta = metadata[document._id]
        #if(received_meta == "Data unavailable"):
            #TODO: Get from Web API
        document.set_meta(received_meta)