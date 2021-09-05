# Document Corpus Generation Pipelines - SciQUACK Document Corpus Creation

This repository contains the code needed to recreate the SciQUACK document corpus and, more generally, to create performant processing pipelines for scientific publications for the creation of a document corpus usable by a [Haystack](https://haystack.deepset.ai/) based Question Answering (QA) System.

__Contents:__

1. Setup
2. Concepts of Document Corpus Generation Pipelines
3. Creating a pipeline
4. Recreating the SciQUACK document corpus
## 1. Setup
To use the functionalities of the document corpus generation pipelines (dcgp) package it has to be installed using the python package installer pip. To install the package clone the repository and execute the following command from the root folder of this repository:

```
pip install ./dcgp
```
If you want to add new components to the package or modify the code in another way you can install the package in editable mode:

```
pip install -e ./dcgp
```

When you have installed the package you can import the package into your python projects:
```python
import dcgp
```

## 2. Concepts of Document Corpus Generation Pipelines
Documents are processed in a pipeline. A pipeline can be represented by an Directed Acyclic Graph. A pipeline is comprised of up to 3 different kinds of pipeline elements. Each pipeline starts with a single pipeline element, a so called Adapter, which feeds data into the pipeline. Next an arbitrary number of so called Processors that modify documents in a predefined way. At the end of a pipeline there are one or more so called Sinks that forward the documents to their desired destination.

### 2.1. Adapters
Adapters feed data into the pipeline. The adapters are specialized for a specific datasource (e.g., plain txt file parses of publications in a folder, documents residing in an elasticsearch documentstore,...). The idea is that the adapter transforms the input to a generalized format (dictionary of text and metadata), independent of the form of the input data source.

### 2.2. Processors
Processors transform documents that pass them. There is a variety of predefined processors of the following categories:

- Splitting Preprocessors
- Metadata Processors that add, remove or modify metadata
- Text Processors that modify the text
- Filter Processors that conditionally discard documents
- Specialized processors that perform complex activities like entity recognition or transformer based classification

### 2.3. Sinks
Sinks direct documents to their desired destination. There are sinks that interact with documentstores (elasticsearch, milvus) and others that interact with the filesystem.

### 2.4. Pipelines
As introduced in the introduction of this chapter pipelines are the core concept of dcgp that combined Adapters, Processors and Sinks. A pipeline is defined sequentially and can be extensively configured. The pipeline handles processing execution, coordinates the different components and provides an easily accessible interface for users.

## 3. Creating a pipeline
Defining a pipeline consists of two actions: Configuring general parameters of the pipeline and building the pipeline by adding pipeline elements to it.

### 3.1. Configuring general parameters
There are two main parameters that can be modified:

1. Whether the pipeline should run in batch mode (processing a batch of documents at each pipeline element at once) or not (processing a single document through the entire pipeline before processing the next document)
2. How many processing cores to use in parallel. *Note: Only the processors are executed in parallel, not the adapter or the sinks to avoid concurrency related issues in reading and writing*

An exemplary configuration with a batch size of 200 documents and using 10 cpu cores in parallel looks the following:
```python
adapter = Adapters.UnarxiveAdapter("../sample")
pipeline = Pipeline(adapter, batch_mode=True, batch_size=200, cpus=10)
```

### 3.2. Building a pipeline
Building a pipeline starts with defining the general pipeline settings and its Adapter, as described in Section 3.1.

Next a number of processors are added:
```python
pipeline.add_processor(Processors.TextKeywordCut("Introduction"))

pipeline.add_processor(Processors.TextReplaceFilter(r"\{{(.*?)\}}", ""))

pipeline.add_processor(Processors.TextSentenceDiscardNonAlpha(0.5))
```

Finally at least one Sink is added to the pipeline:
```python
pipeline.add_sink(Sinks.TextfileSink("./sampleoutput", ["arixiv-id", "_split_id"]))
```

This fully defines the pipeline. Processing can be started thereafter:

```python
pipeline.process()
```

## 4. Recreating the SciQUACK document corpus
To recreate the SciQUACK document corpus you need some additional data:
1. The unarXive dataset that is available on [Zenodo](https://zenodo.org/record/4313164)
2. The arXiv metadata dataset available from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv)
3. The pretrained transformer model for the IMRaD classification task from [Dropbox](https://www.dropbox.com/sh/ze4iupdx78a7ac5/AAAYfvd0XOM87PJuERnKpoiEa?dl=0)

When you have all the additional files you need to prepare the corpus generation and to execute it.

### 4.1. Preparing the corpus generation
The pipeline for the corpus generation is defined in the [generate_sciquack_corpus.py](./scripts/generate_sciquack_corpus.py) script. The location of the datasets, models, etc. mentioned in the following need to be modified in this file.

#### 4.1.1. Prepare the unarXive dataset
Extract the dataset to a folder of your choosing. The relevant folder is called papers and includes full text parses of the papers contained in the unarXive dataset.
#### 4.1.2. Prepare the arXiv metadata
The arXiv metadata is provided in the form of a RESTful disk-based database. [This implementation](https://github.com/daniel-gomm/minimal_RESTful_rocksdb) of a minimalistic RocksDB based database is used.

An instance of this database can be started in a docker container using docker-compose. From the repositories root folder execute:
```
docker-compose -f ./corpus_generation/rocksDB/docker-compose.yml up -d
```
To stop the database run:
```
docker-compose -f ./corpus_generation/rocksDB/docker-compose.yml down
```
Before running the corpus generation you need to add the metadata to the database once. To do this you can use the jupyter notebook [json_to_rocksdb.ipynb](./notebooks/preprocessing/json_to_rocksdb.ipynb). *Note: Make sure to adjust the locations of the metadata file and the url of the database*
#### 4.1.3. Setup a documentstore
You need to setup your desired documentstore the documents should be saved to before starting the corpus generation. Haystack provides a [good overview](https://haystack.deepset.ai/usage/document-store) of the documentstores and how to spin up basic versions of them. For more elaborate instructions on [Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docker.html) and [Milvus](https://milvus.io/docs/v1.1.1/milvus_docker-cpu.md) look at their respective websites.
#### 4.1.4. Customize the pipeline code
Finally you need to adjust the [code of the pipeline](./scripts/generate_sciquack_corpus.py) to match your configuration (in terms of hardware) and to adjust links to files, network locations etc.

### 4.2. Generating the corpus

When you have setup everything correctly you can generate the corpus by executing the following command from the root directory of this repository:
```bash
cd scripts
python generate_sciquack_corpus.py
```

If you have any questions feel free to reach out to us (daniel.gomm@student.kit.edu).
