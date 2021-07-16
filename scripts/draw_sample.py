import argparse
import json

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from pathlib import Path
from kdpipelines import Adapters

def __get_args():
    """Parses the CLI arguments

    Returns
    -------
    Namespace
        Contains all arguments
    """
    parser = argparse.ArgumentParser(
        description = '''Draws sample data from an elasticsearch instance''')    
    parser.add_argument('--host', type=str, default="localhost", help='The server address.')
    parser.add_argument('--out_dir', type=str, default="~/Downloads", help='The directory where you want to store the files.')
    parser.add_argument('--num_samples', type=int, default=5, help='The number of samples you want.')
    
    return parser.parse_args()

def get_sample_from_server(num_samples:int=5, host:str="localhost")->list:
     document_store = ElasticsearchDocumentStore(host=host, username="", password="", index="document")
     adapt = Adapters.ElasticsearchAdapter(document_store, batch_size=1000, filters={"has_datasets":["true"]})
     documents = adapt.generate_documents(num_samples)
     return documents

def write_docs(documents, out_dir):
    count = 0
    for doc in documents:
        with open(Path(out_dir) / (str(count) + '.txt'),'w') as f:
            json.dump(doc, f, indent=4)
            count += 1
    print('{} files written'.format(str(count)))

if __name__ == "__main__":
    """-----Drawing samples from elasticsearch-----"""
    args = __get_args()
    documents = get_sample_from_server(args.num_samples, args.host)
    write_docs(documents, args.out_dir)
