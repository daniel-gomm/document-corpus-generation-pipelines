import requests
import json


class RocksDBAdapter:

    def __init__(self, name:str, url:str):
        """The RocksDBAdapter interacts with a RocksDB instance accessible through a REST interface (https://github.com/daniel-gomm/minimal_RESTful_rocksdb).

        Args:
            name (str): Name of the database to access.
            url (str): URL of the database.
        """
        self._name = name
        self._url = url
        requests.put("{}/single/{}".format(url, self._name))
    
    def put(self, key, value):
        """Add a new key-value pair to the database.

        Args:
            key ([type]): Identifier of a data entity.
            value ([type]): Value of a data entity.
        """
        return requests.post("{}/single/{}/{}".format(self._url, self._name, key), value)
    
    def get(self, key):
        """Retrieves the value for a given key.

        Args:
            key ([type]): Identifier of a data entity. 

        Returns:
            [type]: Value of the entity with the provided key.
        """
        return requests.get("{}/single/{}/{}".format(self._url, self._name, key))
    
    def delete(self, key):
        """Deletes an entity with a given key.

        Args:
            key ([type]): Identifier of a data entity.
        """
        return requests.delete("{}/single/{}/{}".format(self._url, self._name, key))
    
    def update(self, key, value):
        """Updates the value of an entity with a given key.

        Args:
            key ([type]): Identifier of a data entity.
            value ([type]): New value to assign to entity.

        Returns:
            [type]: [description]
        """
        return requests.put("{}/single/{}/{}".format(self._url, self._name, key), value)
    
    def put_all(self, dictionary:dict):
        """Add a list of new key-value pairs to the database.

        Args:
            dictionary (dict): Key-value pairs to be saved to the database.
        """
        return requests.post("{}/multiple/{}".format(self._url, self._name), json = dictionary)
    
    def get_all(self, list_of_keys:list):
        """Retrieves the values for a given list of keys. 

        Args:
            list_of_keys (list): List of keys whichs associated values should be retrived.

        Returns:
            [type]: [description]
        """
        js = json.dumps(list_of_keys)
        headers = {'Content-type': 'application/json'}
        return requests.get("{}/multiple/{}".format(self._url, self._name), data=js, headers=headers)



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
    """Put the contents of a file containing json entities into a RESTful RocksDB database.

    Args:
        filepath (str): Path of the file.
        db_adapter (RocksDBAdapter): Adapter entity.
        chunk_size (int, optional): Chunk size in which entities should be added. Defaults to 100.
        failed_entities_limit (int, optional): Limit of maximum entities that can fail before the process terminates. Defaults to 2000.

    Returns:
        (list, int): The line of the json file in which the process exited and the entities that failed to be added to the database.
    """
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