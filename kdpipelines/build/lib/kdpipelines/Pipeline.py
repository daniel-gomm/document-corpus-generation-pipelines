import sys
import traceback
import time
from re import S
from typing import List, Sized
from .pipeline_elements.Processors import Processor
from .pipeline_elements.Sinks import Sink
from .pipeline_elements.Adapters import Adapter
from multiprocessing import Pool, cpu_count

import logging
logging.basicConfig(filename="../../pipeline_logs.log", level=logging.INFO)

#Helpers

class Progressbar:

    def __init__(self, count:int, prefix:str="", size:int=60, file=sys.stdout):
        self._count = count
        self._size = size
        self._file = file
        self._prefix = prefix
    
    def update(self, position:int, postfix:str=""):
        x = int(self._size*position/self._count)
        self._file.write("%s[%s%s] %i/%i %s\r" % (self._prefix, "#"*x, "."*(self._size-x), position, self._count, postfix))
        self._file.flush()
    

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

    def __init__(self, adapter:Adapter, batch_mode:bool = True, batch_size:int=200, cpus:int=1, max_runtime:int=None):
        """Pipeline that processes documents from an adapter through processors, before outputting the result in sinks.

        Args:
            adapter (Adapter): Adapter that acts as source for the documents.
            batch_mode (bool, optional): Processing mode; when enabled documents are processed in groups (batches). Defaults to True.
            batch_size (int, optional): Size of individual batches. Defaults to 200.
            cpus (int, optional): Number of CPU cores used in parallel for processing (Should be lower than the actual amount of (virtual) cpu cores). Defaults to 1.
            max_runtime(int, optional): Specifies the time in seconds after which processing should be gracefully stopped. If no value is provided the runtime is not limited.

        Raises:
            ValueError: Raised if number of cpus is not properly configured.
        """
        self._max_runtime = max_runtime
        self._processors:List[Processor] = []
        self._sinks:List[Processor] = []
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._adapter = adapter
        self._processed_documents = 0
        self._output_documents = 0
        if cpus > cpu_count() or cpus < 1:
            raise ValueError(f"Number of processors exceedes the machines capabilities (max. {cpu_count()} processors)")
        self._cpus = cpus
    
    def add_processor(self, processor:Processor):
        """Add a processor to the Pipeline.

        Args:
            processor (Processor): Configured Processor.
        """        
        self._processors.append(processor)
    
    def add_sink(self, sink:Sink):
        """Add a sink to the Pipeline.

        Args:
            sink (Sink): Configured Sink.
        """        
        self._sinks.append(sink)
    
    def process(self):
        """The documents from the adapter are processed according to the Pipelines configuration
        """        
        logging.info(f"Pipeline processing started. A total of {len(self._adapter)} documents will be processed.")
        if not self._batch_mode:
            self._batch_size = 1
        if self._cpus > 1:
            self._process_multicore()
        else:
            self._process_singlecore()
    
    def _process_multicore(self):
        end_time = None
        if self._max_runtime:
            end_time = time.time() + self._max_runtime
        with Pool(self._cpus) as pool:
            documentProcessor = DocumentProcessor(self._processors)
            cycles = int(len(self._adapter)/(self._batch_size*self._cpus))+1
            bar = Progressbar(cycles, f"Processing batches ({self._cpus} minibatches at once): ")
            for i in range(cycles):
                bar.update(i, f"Processed documents: {self._processed_documents} Output Documents: {self._output_documents}")
                docs_batches = []
                batches_listed = 0
                while batches_listed < self._cpus and len(self._adapter) > 0:
                    docs_batches.append(self._adapter.generate_documents(self._batch_size))
                    batches_listed += 1
                for batch in docs_batches:
                    self._processed_documents += len(batch)
                #Process a batch of documents on each processing core
                results = pool.map(documentProcessor.process_docs, docs_batches)
                #Process results in sinks sequentially
                for result in results:
                    self._output_documents += len(result)
                    self._process_sinks(result)
                if end_time:
                    if time.time() > end_time:
                        logging.info("Maximum processing time exceeded. Stopping the pipeline.")
                        break
            bar.update(cycles, f"Processed documents: {self._processed_documents} Output Documents: {self._output_documents}\nProcessing Done!")

    def _process_singlecore(self):
        documentProcessor = DocumentProcessor(self._processors)
        end_time = None
        if self._max_runtime:
            end_time = time.time() + self._max_runtime
        cycles = int(len(self._adapter)/self._batch_size)+1
        bar = Progressbar(cycles, "Processing batches: ")
        for i in range(cycles):
            bar.update(i, f"Processed documents: {self._processed_documents} Output Documents: {self._output_documents}")
            if len(self._adapter) > 0:
                docs = self._adapter.generate_documents(self._batch_size)
                self._processed_documents += len(docs)
                docs = documentProcessor.process_docs(docs)
                self._output_documents += len(docs)
                self._process_sinks(docs)
            if end_time:
                if time.time() > end_time:
                    logging.info("Maximum processing time exceeded. Stopping the pipeline.")
                    break
        bar.update(cycles, f"Processed documents: {self._processed_documents} Output Documents: {self._output_documents}\nProcessing Done!")
    
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
