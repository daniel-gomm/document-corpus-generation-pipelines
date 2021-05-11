from arxive_metadata import rocksDB
import re
import os
import unarxive_import_helpers as helpers
import sys, getopt

def main(argv):
    try:
      opts, args = getopt.getopt(argv,"hp:u:",["path=","url="])
    except getopt.GetoptError:
      print('Please specify the path of the unarXive data and the RocksDB url:\n\nImport_unarxive.py --path <path_to_unarxive_directory> --url <url_of_rocksdb>')
      sys.exit(2)
    db_url = None
    directory = None
    for opt, arg in opts:
      if opt == '-h':
         print('Please specify the path of the unarXive data and the RocksDB url:\n\nImport_unarxive.py --path <path_to_unarxive_directory> --url <url_of_rocksdb>')
         sys.exit()
      elif opt in ("-p", "--path"):
         directory = arg
      elif opt in ("-u", "--url"):
         db_url = arg
    if (db_url is None or directory is None):
        print('Please specify the path of the unarXive data and the RocksDB url:\n\nImport_unarxive.py --path <path_to_unarxive_directory> --url <url_of_rocksdb>')
        sys.exit(2)
    #All unarXive plain text data is saved in "directory"
    #directory = "/home/daniel/KD/samples"
    files = os.listdir(directory)
    #Open the directory and list all textfiles
    txt_files = list(filter(lambda x: x.endswith(".txt"), files))
    #Initialize the database holding the metadata information
    #url = "http://192.168.178.34:8089"
    db = rocksDB.RocksDBAdapter("arXive_metadata", db_url)
    #Process the files. Combine them with their respective metadata and return them as list.
    statistics = helpers.process_docements(directory, txt_files, db, 400)
    print("Processed {} documents in {} seconds.\nA total amount of {} documents was found.\nNo metadata was found for {} documents.".format(statistics["no_original_papers"], statistics["elapsed_time"], statistics["no_output_papers"], statistics["no_paper_without_metadata"]))


if __name__ == "__main__":
   main(sys.argv[1:])