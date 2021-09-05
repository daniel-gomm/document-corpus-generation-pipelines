from dcgp.Pipeline import Pipeline
from dcgp import Adapters, Processors, Sinks
import json

#Define adapter for the unarXive dataset
adapter = Adapters.UnarxiveAdapter("/media/daniel/01D6FA6DDBE9CC00/unarxive/unarXive-2020/papers")
print(f"A total of {len(adapter)} documents will be processed.")
pipeline = Pipeline(adapter, batch_size=100, cpus=10)

#Add Metadata from arXiv.org to each document
from dcgp.database.rocksDB import RocksDBAdapter
db = RocksDBAdapter("arXive_metadata", "http://localhost:8089")
pipeline.add_processor(Processors.MetadataArxiveEnricher("arixive-id",db))

#Discard all documents that are not concerned with the computer science domain
cs_values = ['cs.AI', 'cs.AR', 'cs.CE', 'cs.CL', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.GL', 'cs.HC', 'cs.IR', 'cs.LG', 'cs.MA', 'cs.NE', 'cs.NI', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SE', 'cs.SI', 'cs.SY']
#cs_values = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR","cs.CV","cs.CY","cs.DB","cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR", "cs.GT","cs.HC","cs.IR","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI","cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY"]
pipeline.add_processor(Processors.FilterOnMetadataValue("categories", cs_values))

#Clean up text
#Remove everything in front of the introduction
pipeline.add_processor(Processors.TextKeywordCut("Introduction"))
#Remove everything beyond the Acknowledgements
pipeline.add_processor(Processors.TextKeywordCut("Acknowledgments", False))
#Add the abstract to the beginning of the paper
pipeline.add_processor(Processors.TextAppendMetadataField("abstract", True, line_mode=True))

#Discard unnecessary metadata fields
fields_to_discard = ["update_date", "doi", "authors", "abstract", "versions", "comments", "report-no", "license", "id", "categories", "submitter", "journal-ref", "authors_parsed"]
pipeline.add_processor(Processors.MetadataFieldDiscarder(fields_to_discard))

#Add the makg id of the paper
import pandas as pd
df = pd.read_csv("../data/unarxive_metadata/mag_id_2_arxiv_id.csv", names=["mag_id", "idk", "url", "arxive_id"])
pipeline.add_processor(Processors.MetadataMagArxiveLinker(df, column_to_match="arxive_id", column_to_add="mag_id", field_to_match="arixive-id"))
del df

#Extract paragraphs to lines
pipeline.add_processor(Processors.LineUnarxiveGenerator())


#TextReplaceFilter
#Remove unarxive references
pipeline.add_processor(Processors.TextReplaceFilter(r"\{{(.*?)\}}", ""))
#Remove multiple commata
pipeline.add_processor(Processors.TextReplaceFilter(r" ,", ""))
#Remove (REF )
pipeline.add_processor(Processors.TextReplaceFilter(r"\(REF(.*?)\)", ""))
#Remove excessive whitespace
pipeline.add_processor(Processors.TextReplaceFilter(r"[ ]{2,}", " "))
#Remove excessive whitespace
pipeline.add_processor(Processors.TextReplaceFilter(r"FIGURE", " "))

#Remove sentences containing predominantely nonalphanumeric words or Formulas
pipeline.add_processor(Processors.TextSentenceDiscardNonAlpha(0.5))
pipeline.add_processor(Processors.TextSentenceDiscardFilter("FORMULA", 0.2))

#Discard extremely short paper
pipeline.add_processor(Processors.FilterShortDocuments(500))

#Add status field to metadata (only for maintenance)
pipeline.add_processor(Processors.MetadataFieldAdder("_status", 0))

#Custom pre processor that splits the document respecting sentences and paragraphs
pipeline.add_processor(Processors.SplitByLinePreProcessor(split_len = 100, minimal_split_len = 70))

#Annotate found datasets in the publication
with open('../data/pwc_metadata/datasets_preprocessed.json') as f:
    datasets = json.load(f)
pipeline.add_processor(Processors.StringMatchingProcessor(datasets, info_field_name='datasets', info_key='pwc_url'))

#Annotate found methods in the publication
with open('../data/pwc_metadata/methods_preprocessed.json') as f:
    methods = json.load(f)
pipeline.add_processor(Processors.StringMatchingProcessor(methods, info_field_name='methods', info_key='pwc_url'))

#Add IMRaD Classification
pipeline.add_processor(Processors.MetadataFieldAdder("Section", ""))
model_path='/home/daniel/KIT/KD/ScibertUncased5.model'
pipeline.add_processor(Processors.IMRaDClassification(model_path))

#ElasticsearchSink - store the processed documents to an elasticsearch instance
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")
pipeline.add_sink(Sinks.ElasticsearchSink(document_store))

#CSVMetadataSink - saves the ids of the added documents
pipeline.add_sink(Sinks.CSVMetadataSink("./added_mag_ids.csv", "mag_id"))

print(pipeline)
#Start processing
pipeline.process()