import abc
import sys
from re import S
from typing import List
from Processors import Processor
from Sinks import Sink
from Adapters import Adapter

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

class Pipeline:

    def __init__(self, adapter:Adapter, batch_mode:bool = True, batch_size:int=200):
        self._processors:List[Processor] = []
        self._sinks:List[Processor] = []
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._adapter = adapter
    
    def add_processor(self, processor:Processor):
        self._processors.append(processor)
    
    def add_processors(self, processors:List[Processor]):
        self._processors.append(processors)
    
    def add_sink(self, sink:Sink):
        self._sinks.append(sink)
    
    def process(self):
        if not self._batch_mode:
            self._batch_size = 1
        for i in progressbar(range(int(len(self._adapter)/self._batch_size)+1), "Processing batches: "):
            if len(self._adapter) > 0:
                docs = self._adapter.generate_documents(self._batch_size)
                for processor in self._processors:
                    docs = processor.process(docs)
                for sink in self._sinks:
                    sink.process(docs)
    
    def __str__(self) -> str:
        ret = "({})".format(type(self._adapter).__name__)
        for processor in self._processors:
            ret += " ---> " + "*{}*".format(type(processor).__name__)
        for sink in self._sinks:
            ret += "\n===> " + "|_{}_|".format(type(sink).__name__)
        return ret
