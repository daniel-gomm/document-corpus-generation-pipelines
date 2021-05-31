import abc
import sys
import traceback
from re import S
from typing import List
from Processors import Processor
from Sinks import Sink
from Adapters import Adapter
from multiprocessing import Pool, cpu_count

import logging
logging.basicConfig(filename="pipeline_logs.log", level=logging.INFO)

#Helpers

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def process_docs_through_processors(documents:List[dict], processors:List[Processor]) -> List:
    for processor in processors:
        if documents:
            documents = processor.process(documents)
    return documents

class DocumentProcessor:

    def __init__(self, processors:List[Processor]) -> None:
        self._processors = processors

    def process_docs(self, documents:List[dict]):
        for processor in self._processors:
            try:
                if documents:
                    documents = processor.process(documents)
            except:
                logging.error(traceback.format_exc())
        return documents

class Pipeline:

    def __init__(self, adapter:Adapter, batch_mode:bool = True, batch_size:int=200, cpus:int=1):
        self._processors:List[Processor] = []
        self._sinks:List[Processor] = []
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._adapter = adapter
        if cpus > cpu_count() or cpus < 1:
            raise ValueError(f"Number of processors exceedes the machines capabilities (max. {cpu_count()} processors)")
        self._cpus = cpus
    
    def add_processor(self, processor:Processor):
        self._processors.append(processor)
    
    def add_processors(self, processors:List[Processor]):
        self._processors.append(processors)
    
    def add_sink(self, sink:Sink):
        self._sinks.append(sink)
    
    def process(self):
        logging.info(f"Pipeline processing started. A total of {len(self._adapter)} documents will be processed.")
        if not self._batch_mode:
            self._batch_size = 1
        if self._cpus > 1:
            self._process_multicore()
        else:
            self._process_singlecore()
    
    def _process_multicore(self):
        with Pool(self._cpus) as pool:
            documentProcessor = DocumentProcessor(self._processors)
            for i in progressbar(range(int(len(self._adapter)/(self._batch_size*self._cpus))+1), f"Processing batches ({self._cpus} minibatches at once): "):
                docs_batches = []
                batches_listed = 0
                while batches_listed < self._cpus and len(self._adapter) > 0:
                    docs_batches.append(self._adapter.generate_documents(self._batch_size))
                    batches_listed += 1
                #Process a batch of documents on each processing core
                results = pool.map(documentProcessor.process_docs, docs_batches)
                #Process results in sinks sequentially
                for result in results:
                    self._process_sinks(result)

    def _process_singlecore(self):
        for i in progressbar(range(int(len(self._adapter)/self._batch_size)+1), "Processing batches: "):
            if len(self._adapter) > 0:
                docs = self._adapter.generate_documents(self._batch_size)
                docs = self._process_processors(docs)
                self._process_sinks(docs)
    
    def _process_sinks(self, documents:List):
        for sink in self._sinks:
            try:
                if documents:
                    sink.process(documents)
            except:
                logging.error(traceback.format_exc())
    
    def __str__(self) -> str:
        ret = f"({type(self._adapter).__name__})"
        for processor in self._processors:
            ret += f" ---> *{type(processor).__name__}*"
        for sink in self._sinks:
            ret += f"\n===> |_{type(sink).__name__}_|"
        return ret
