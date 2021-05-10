import requests
import json


class RocksDBAdapter:
    
    def __init__(self, name:str, url:str):
        self._name = name
        self._url = url
        requests.put("{}/single/{}".format(url, self._name))
    
    def put(self, key, value):
        return requests.post("{}/single/{}/{}".format(self._url, self._name, key), value)
    
    def get(self, key):
        return requests.get("{}/single/{}/{}".format(self._url, self._name, key))
    
    def delete(self, key):
        return requests.delete("{}/single/{}/{}".format(self._url, self._name, key))
    
    def update(self, key, value):
        return requests.put("{}/single/{}/{}".format(self._url, self._name, key), value)
    
    def put_all(self, dictionary:dict):
        return requests.post("{}/multiple/{}".format(self._url, self._name), json = dictionary)
    
    def get_all(self, list_of_keys:list):
        return requests.get("{}/multiple/{}".format(self._url, self._name), json = list_of_keys)



def extract_id(json_entry:str):
    meta_dict = json.loads(json_entry)
    return meta_dict["id"]


def handle_response(response:requests.Response):
    if(response.ok):
        print("Successfully put entries into database.")
        return True
    else:
        print("Failed to put entries into database.")
        return False


def file_to_db(filepath:str, db_adapter:RocksDBAdapter, chunk_size:int=100, failed_entities_limit:int=2000):
    with open(filepath, "rt") as metadata_file:
        counter = 0
        dictionary = {}
        failed_entities = {}
        for line_no, line in enumerate(metadata_file):
            counter = counter + 1
            line = line[:-1]
            dictionary[extract_id(line)] = line
            if (counter >= chunk_size):
                counter = 0
                response = db_adapter.put_all(dictionary)
                if(not handle_response(response)):
                    failed_entities.update(dictionary)
                    if (failed_entities.__len__() > failed_entities_limit):
                        print("Limit of failed entities reached. Exiting...")
                        return failed_entities, line_no
                dictionary.clear()
                print("{} entries have been added to the database".format(line_no-failed_entities.__len__()))
        if(dictionary.__len__() > 0):
            response = db_adapter.put_all(dictionary)
            if(not handle_response(response)):
                failed_entities.update(dictionary)
        print("Finished successfully.")
        return failed_entities, line_no