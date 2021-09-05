from dcgp.Pipeline import Pipeline
from dcgp import Adapters, Processors, Sinks

adapter = Adapters.UnarxiveAdapter("/media/daniel/01D6FA6DDBE9CC00/unarxive/unarXive-2020/papers")
#adapter = Adapters.UnarxiveAdapter("/home/daniel/KIT/KD/samples/sample", line_mode=True)
print(f"A total of {len(adapter)} documents will be processed.")
pipeline = Pipeline(adapter, batch_size=100, cpus=10)

#MetadataArxiveEnricher
from arxive_metadata.rocksDB import RocksDBAdapter
db = RocksDBAdapter("arXive_metadata", "http://localhost:8089")
pipeline.add_processor(Processors.MetadataArxiveEnricher("arixive-id",db))

#FilterOnMetadataValue
cs_values = ['cs.AI', 'cs.AR', 'cs.CE', 'cs.CL', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.GL', 'cs.HC', 'cs.IR', 'cs.LG', 'cs.MA', 'cs.NE', 'cs.NI', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SE', 'cs.SI', 'cs.SY']
#All cs categories
#cs_values = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR","cs.CV","cs.CY","cs.DB","cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL","cs.GL","cs.GR", "cs.GT","cs.HC","cs.IR","cs.LG","cs.LO","cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI","cs.OH","cs.OS","cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY"]
pipeline.add_processor(Processors.FilterOnMetadataValue("categories", cs_values))

#Clean up text
#TextKeywordCut Introduction
pipeline.add_processor(Processors.TextKeywordCut("Introduction"))
#TextKeywordCut Acknowledgements
pipeline.add_processor(Processors.TextKeywordCut("Acknowledgments", False))
#TextAppendMetadataField
pipeline.add_processor(Processors.TextAppendMetadataField("abstract", True, line_mode=True))

#MetadataFieldDiscarder
fields_to_discard = ["update_date", "doi", "authors", "abstract", "versions", "comments", "report-no", "title", "license", "id", "categories", "submitter", "journal-ref", "authors_parsed"]
pipeline.add_processor(Processors.MetadataFieldDiscarder(fields_to_discard))

#MetadataMagArxiveLinker
import pandas as pd
df = pd.read_csv("/home/daniel/KIT/KD/kd-documents/data/unarxive_metadata/mag_id_2_arxiv_id.csv", names=["mag_id", "idk", "url", "arxive_id"])
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

#SentenceLevelProcessors
pipeline.add_processor(Processors.TextSentenceDiscardNonAlpha(0.5))
pipeline.add_processor(Processors.TextSentenceDiscardFilter("FORMULA", 0.2))

#Discard extremely short paper
pipeline.add_processor(Processors.FilterShortDocuments(500))

#Add status field to metadata
pipeline.add_processor(Processors.MetadataFieldAdder("_status", 0))

'''#HaystackPreProcessor
from haystack.preprocessor import PreProcessor
pre_processor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by='word',
    split_length=100,
    split_respect_sentence_boundary=True,
    split_overlap=0)
pipeline.add_processor(Processors.HaystackPreProcessor(pre_processor))'''
#Custom pre processor that splits the document respecting sentences and paragraphs
pipeline.add_processor(Processors.SplitByLinePreProcessor(split_len = 100, minimal_split_len = 70))

#ElasticsearchSink
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="aifb-ls3-vm1.aifb.kit.edu", username="", password="", index="document")
pipeline.add_sink(Sinks.ElasticsearchSink(document_store))

#TextfileSink
#pipeline.add_sink(Sinks.TextfileSink("/home/daniel/KIT/KD/samples/testoutput", ["arixive-id", "_split_id"]))
#pipeline.add_sink(Sinks.TextfileSink("/home/daniel/KIT/KD/samples/testoutput", ["arixive-id"]))
#CSVMetadataSink
pipeline.add_sink(Sinks.CSVMetadataSink("/home/daniel/KIT/KD/added_mag_ids.csv", "mag_id"))

print(pipeline)

pipeline.process()