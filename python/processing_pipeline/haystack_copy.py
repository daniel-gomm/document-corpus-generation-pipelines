import json
import logging
import time
from copy import deepcopy
from string import Template
from typing import List, Optional, Union, Dict, Any, Generator
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch.helpers import bulk, scan
from elasticsearch.exceptions import RequestError
import numpy as np
from scipy.special import expit
from tqdm.auto import tqdm
from pathlib import Path
from abc import abstractmethod

logger = logging.getLogger(__name__)


from typing import Any, Optional, Dict, List
from uuid import uuid4
import mmh3

class DuplicateDocumentError(ValueError):
    """Exception for Duplicate document"""
    pass


class BaseComponent:
    """
    A base class for implementing nodes in a Pipeline.
    """

    outgoing_edges: int
    subclasses: dict = {}
    pipeline_config: dict = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific component implementations.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """
        Load a component instance of the given type using the kwargs.
        
        :param component_type: name of the component class to load.
        :param kwargs: parameters to pass to the __init__() for the component. 
        """
        if component_type not in cls.subclasses.keys():
            raise Exception(f"Haystack component with the name '{component_type}' does not exist.")
        instance = cls.subclasses[component_type](**kwargs)
        return instance

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any):
        """
        Method that will be executed when the node in the graph is called.
        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)
        See an example for an implementation in haystack/reader/base/BaseReader.py
        :param kwargs:
        :return:
        """
        pass

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.

        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v

class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        score: Optional[float] = None,
        probability: Optional[float] = None,
        question: Optional[str] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id_hash_keys: Optional[List[str]] = None
    ):
        """
        One of the core data classes in Haystack. It's used to represent documents / passages in a standardized way within Haystack.
        Documents are stored in DocumentStores, are returned by Retrievers, are the input for Readers and are used in
        many other places that manipulate or interact with document-level data.

        Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
        into smaller passages. We'll have one Document per passage in this case.

        Each document has a unique ID. This can be supplied by the user or generated automatically.
        It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

        There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

        :param text: Text of the document
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: Retriever's query score for a retrieved document
        :param probability: a pseudo probability by scaling score in the range 0 to 1
        :param question: Question text (e.g. for FAQs where one document usually consists of one question and one answer text).
        :param meta: Meta fields for a document like name, url, or author.
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].
        """

        self.text = text
        self.score = score
        self.probability = probability
        self.question = question
        self.meta = meta or {}
        self.embedding = embedding

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = self._get_id(id_hash_keys)

    def _get_id(self, id_hash_keys):
        final_hash_key = ":".join(id_hash_keys) if id_hash_keys else self.text
        return '{:02x}'.format(mmh3.hash128(final_hash_key, signed=False))

    def to_dict(self, field_map={}):
        inv_field_map = {v:k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}):
        _doc = dict.copy()
        init_args = ["text", "id", "score", "probability", "question", "meta", "embedding"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


class Label:
    def __init__(self, question: str,
                 answer: str,
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: str,
                 id: Optional[str] = None,
                 document_id: Optional[str] = None,
                 offset_start_in_doc: Optional[int] = None,
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None,
                 created_at: Optional[str] = None,
                 updated_at: Optional[str] = None,
                 meta: Optional[dict] = None
                 ):
        """
        Object used to represent label/feedback in a standardized way within Haystack.
        This includes labels from dataset like SQuAD, annotations from labeling tools,
        or, user-feedback from the Haystack REST API.

        :param question: the question(or query) for finding answers.
        :param answer: the answer string.
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param id: Unique ID used within the DocumentStore. If not supplied, a uuid will be generated automatically.
        :param document_id: the document_store's ID for the returned answer document.
        :param offset_start_in_doc: the answer start offset in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        :param created_at: Timestamp of creation with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S").
        :param created_at: Timestamp of update with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S")
        """

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        self.created_at = created_at
        self.updated_at = updated_at
        self.question = question
        self.answer = answer
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.document_id = document_id
        self.offset_start_in_doc = offset_start_in_doc
        self.no_answer = no_answer
        self.model_id = model_id
        if not meta:
            self.meta = dict()
        else:
            self.meta = meta

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    # define __eq__ and __hash__ functions to deduplicate Label Objects
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                getattr(other, 'question', None) == self.question and
                getattr(other, 'answer', None) == self.answer and
                getattr(other, 'is_correct_answer', None) == self.is_correct_answer and
                getattr(other, 'is_correct_document', None) == self.is_correct_document and
                getattr(other, 'origin', None) == self.origin and
                getattr(other, 'document_id', None) == self.document_id and
                getattr(other, 'offset_start_in_doc', None) == self.offset_start_in_doc and
                getattr(other, 'no_answer', None) == self.no_answer and
                getattr(other, 'model_id', None) == self.model_id and
                getattr(other, 'created_at', None) == self.created_at and
                getattr(other, 'updated_at', None) == self.updated_at)

    def __hash__(self):
        return hash(self.question +
                    self.answer +
                    str(self.is_correct_answer) +
                    str(self.is_correct_document) +
                    str(self.origin) +
                    str(self.document_id) +
                    str(self.offset_start_in_doc) +
                    str(self.no_answer) +
                    str(self.model_id)
                    )

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))

class MultiLabel:
    def __init__(self, question: str,
                 multiple_answers: List[str],
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: str,
                 multiple_document_ids: List[Any],
                 multiple_offset_start_in_docs: List[Any],
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None,
                 meta: dict = None
                 ):
        """
        Object used to aggregate multiple possible answers for the same question
        :param question: the question(or query) for finding answers.
        :param multiple_answers: list of possible answer strings
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param multiple_document_ids: the document_store's IDs for the returned answer documents.
        :param multiple_offset_start_in_docs: the answer start offsets in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        """
        self.question = question
        self.multiple_answers = multiple_answers
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.multiple_document_ids = multiple_document_ids
        self.multiple_offset_start_in_docs = multiple_offset_start_in_docs
        self.no_answer = no_answer
        self.model_id = model_id
        if not meta:
            self.meta = dict()
        else:
            self.meta = meta

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())









class BaseDocumentStore(BaseComponent):
    """
    Base class for implementing Document Stores.
    """
    index: Optional[str]
    label_index: Optional[str]
    similarity: Optional[str]
    duplicate_documents_options: tuple = ('skip', 'overwrite', 'fail')

    @abstractmethod
    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000, duplicate_documents: Optional[str] = None):
        """
        Indexes documents for later queries.
        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(
            self,
            index: Optional[str] = None,
            filters: Optional[Dict[str, List[str]]] = None,
            return_embedding: Optional[bool] = None
    ) -> List[Document]:
        """
        Get documents from the document store.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        """
        pass

    @abstractmethod
    def get_all_labels(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]:
        pass

    def get_all_labels_aggregated(self,
                                  index: Optional[str] = None,
                                  filters: Optional[Dict[str, List[str]]] = None,
                                  open_domain: bool=True,
                                  aggregate_by_meta: Optional[Union[str, list]]=None) -> List[MultiLabel]:
        """
        Return all labels in the DocumentStore, aggregated into MultiLabel objects. 
        This aggregation step helps, for example, if you collected multiple possible answers for one question and you
        want now all answers bundled together in one place for evaluation.
        How they are aggregated is defined by the open_domain and aggregate_by_meta parameters.
        If the questions are being asked to a single document (i.e. SQuAD style), you should set open_domain=False to aggregate by question and document.
        If the questions are being asked to your full collection of documents, you should set open_domain=True to aggregate just by question.
        If the questions are being asked to a subslice of your document set (e.g. product review use cases),
        you should set open_domain=True and populate aggregate_by_meta with the names of Label meta fields to aggregate by question and your custom meta fields.
        For example, in a product review use case, you might set aggregate_by_meta=["product_id"] so that Labels
        with the same question but different answers from different documents are aggregated into the one MultiLabel
        object, provided that they have the same product_id (to be found in Label.meta["product_id"])
        :param index: Name of the index to get the labels from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the labels to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param open_domain: When True, labels are aggregated purely based on the question text alone.
                            When False, labels are aggregated in a closed domain fashion based on the question text
                            and also the id of the document that the label is tied to. In this setting, this function
                            might return multiple MultiLabel objects with the same question string.
        :param aggregate_by_meta: The names of the Label meta fields by which to aggregate. For example: ["product_id"]
        """
        aggregated_labels = []
        all_labels = self.get_all_labels(index=index, filters=filters)

        # Collect all answers to a question in a dict
        question_ans_dict: dict = {}
        for l in all_labels:
            # This group_by_id determines the key by which we aggregate labels. Its contents depend on
            # whether we are in an open / closed domain setting,
            # or if there are fields in the meta data that we should group by (set using group_by_meta)
            group_by_id_list: list = []
            if open_domain:
                group_by_id_list = [l.question]
            else:
                group_by_id_list = [l.document_id, l.question]
            if aggregate_by_meta:
                if type(aggregate_by_meta) == str:
                    aggregate_by_meta = [aggregate_by_meta]
                for meta_key in aggregate_by_meta:
                    curr_meta = l.meta.get(meta_key, None)
                    if curr_meta:
                        group_by_id_list.append(curr_meta)
            group_by_id = tuple(group_by_id_list)

            # only aggregate labels with correct answers, as only those can be currently used in evaluation
            if not l.is_correct_answer:
                continue

            if group_by_id in question_ans_dict:
                question_ans_dict[group_by_id].append(l)
            else:
                question_ans_dict[group_by_id] = [l]

        # Aggregate labels
        for q, ls in question_ans_dict.items():
            ls = list(set(ls))  # get rid of exact duplicates
            # check if there are both text answer and "no answer" present
            t_present = False
            no_present = False
            no_idx = []
            for idx, l in enumerate(ls):
                if len(l.answer) == 0:
                    no_present = True
                    no_idx.append(idx)
                else:
                    t_present = True
            # if both text and no answer are present, remove no answer labels
            if t_present and no_present:
                logger.warning(
                    f"Both text label and 'no answer possible' label is present for question: {ls[0].question}")
                for remove_idx in no_idx[::-1]:
                    ls.pop(remove_idx)

            # construct Aggregated_label
            for i, l in enumerate(ls):
                if i == 0:
                    # Keep only the label metadata that we are aggregating by
                    if aggregate_by_meta:
                        meta_new = {k: v for k, v in l.meta.items() if k in aggregate_by_meta}
                    else:
                        meta_new = {}

                    agg_label = MultiLabel(question=l.question,
                                           multiple_answers=[l.answer],
                                           is_correct_answer=l.is_correct_answer,
                                           is_correct_document=l.is_correct_document,
                                           origin=l.origin,
                                           multiple_document_ids=[l.document_id],
                                           multiple_offset_start_in_docs=[l.offset_start_in_doc],
                                           no_answer=l.no_answer,
                                           model_id=l.model_id,
                                           meta=meta_new)
                else:
                    agg_label.multiple_answers.append(l.answer)
                    agg_label.multiple_document_ids.append(l.document_id)
                    agg_label.multiple_offset_start_in_docs.append(l.offset_start_in_doc)
            aggregated_labels.append(agg_label)
        return aggregated_labels

    @abstractmethod
    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[Optional[Dict[str, List[str]]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        pass

    @abstractmethod
    def get_label_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def write_labels(self, labels: Union[List[Label], List[dict]], index: Optional[str] = None):
        pass

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        pass

    @abstractmethod
    def delete_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        pass

    def run(self, documents: List[dict], index: Optional[str] = None, **kwargs):  # type: ignore
        self.write_documents(documents=documents, index=index)
        return kwargs, "output_1"

    @abstractmethod
    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None,
                            batch_size: int = 10_000) -> List[Document]:
        pass

    def _drop_duplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
         Drop duplicates documents based on same hash ID
         :param documents: A list of Haystack Document objects.
         :return: A list of Haystack Document objects.
        """
        _hash_ids: list = []
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.warning(f"Duplicate Documents: Document with id '{document.id}' already exists in index "
                               f"'{self.index}'")
                continue
            _documents.append(document)
            _hash_ids.append(document.id)

        return _documents

    def _handle_duplicate_documents(self, documents: List[Document], duplicate_documents: Optional[str] = None):
        """
        Handle duplicates documents
        :param documents: A list of Haystack Document objects.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :return: A list of Haystack Document objects.
       """
        if duplicate_documents in ('skip', 'fail'):
            documents = self._drop_duplicate_documents(documents)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=self.index)
            ids_exist_in_db = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == 'fail':
                raise DuplicateDocumentError(f"Document with ids '{', '.join(ids_exist_in_db)} already exists"
                                             f" in index = '{self.index}'.")

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents















class ElasticsearchDocumentStore(BaseComponent):
    def __init__(
        self,
        host: Union[str, List[str]] = "localhost",
        port: Union[int, List[int]] = 9200,
        username: str = "",
        password: str = "",
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        aws4auth = None,
        index: str = "document",
        label_index: str = "label",
        search_fields: Union[str, list] = "text",
        text_field: str = "text",
        name_field: str = "name",
        embedding_field: str = "embedding",
        embedding_dim: int = 768,
        custom_mapping: Optional[dict] = None,
        excluded_meta_data: Optional[list] = None,
        faq_question_field: Optional[str] = None,
        analyzer: str = "standard",
        scheme: str = "http",
        ca_certs: Optional[str] = None,
        verify_certs: bool = True,
        create_index: bool = True,
        refresh_type: str = "wait_for",
        similarity="dot_product",
        timeout=30,
        return_embedding: bool = False,
        duplicate_documents: str = 'overwrite',
    ):
        """
        A DocumentStore using Elasticsearch to store and query the documents for our search.
            * Keeps all the logic to store and query documents from Elastic, incl. mapping of fields, adding filters or boosts to your queries, and storing embeddings
            * You can either use an existing Elasticsearch index or create a new one via haystack
            * Retrievers operate on top of this DocumentStore to find the relevant documents for a query
        :param host: url(s) of elasticsearch nodes
        :param port: port(s) of elasticsearch nodes
        :param username: username (standard authentication via http_auth)
        :param password: password (standard authentication via http_auth)
        :param api_key_id: ID of the API key (altenative authentication mode to the above http_auth)
        :param api_key: Secret value of the API key (altenative authentication mode to the above http_auth)
        :param aws4auth: Authentication for usage with aws elasticsearch (can be generated with the requests-aws4auth package)
        :param index: Name of index in elasticsearch to use for storing the documents that we want to search. If not existing yet, we will create one.
        :param label_index: Name of index in elasticsearch to use for storing labels. If not existing yet, we will create one.
        :param search_fields: Name of fields used by ElasticsearchRetriever to find matches in the docs to our incoming query (using elastic's multi_match query), e.g. ["title", "full_text"]
        :param text_field: Name of field that might contain the answer and will therefore be passed to the Reader Model (e.g. "full_text").
                           If no Reader is used (e.g. in FAQ-Style QA) the plain content of this field will just be returned.
        :param name_field: Name of field that contains the title of the the doc
        :param embedding_field: Name of field containing an embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param embedding_dim: Dimensionality of embedding vector (Only needed when using a dense retriever (e.g. DensePassageRetriever, EmbeddingRetriever) on top)
        :param custom_mapping: If you want to use your own custom mapping for creating a new index in Elasticsearch, you can supply it here as a dictionary.
        :param analyzer: Specify the default analyzer from one of the built-ins when creating a new Elasticsearch Index.
                         Elasticsearch also has built-in analyzers for different languages (e.g. impacting tokenization). More info at:
                         https://www.elastic.co/guide/en/elasticsearch/reference/7.9/analysis-analyzers.html
        :param excluded_meta_data: Name of fields in Elasticsearch that should not be returned (e.g. [field_one, field_two]).
                                   Helpful if you have fields with long, irrelevant content that you don't want to display in results (e.g. embedding vectors).
        :param scheme: 'https' or 'http', protocol used to connect to your elasticsearch instance
        :param ca_certs: Root certificates for SSL: it is a path to certificate authority (CA) certs on disk. You can use certifi package with certifi.where() to find where the CA certs file is located in your machine.
        :param verify_certs: Whether to be strict about ca certificates
        :param create_index: Whether to try creating a new index (If the index of that name is already existing, we will just continue in any case
        :param refresh_type: Type of ES refresh used to control when changes made by a request (e.g. bulk) are made visible to search.
                             If set to 'wait_for', continue only after changes are visible (slow, but safe).
                             If set to 'false', continue directly (fast, but sometimes unintuitive behaviour when docs are not immediately available after ingestion).
                             More info at https://www.elastic.co/guide/en/elasticsearch/reference/6.8/docs-refresh.html
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default sine it is
                           more performant with DPR embeddings. 'cosine' is recommended if you are using a Sentence BERT model.
        :param timeout: Number of seconds after which an ElasticSearch request times out.
        :param return_embedding: To return document embedding
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(
            host=host, port=port, username=username, password=password, api_key_id=api_key_id, api_key=api_key,
            aws4auth=aws4auth, index=index, label_index=label_index, search_fields=search_fields, text_field=text_field,
            name_field=name_field, embedding_field=embedding_field, embedding_dim=embedding_dim,
            custom_mapping=custom_mapping, excluded_meta_data=excluded_meta_data, analyzer=analyzer, scheme=scheme,
            ca_certs=ca_certs, verify_certs=verify_certs, create_index=create_index,
            duplicate_documents=duplicate_documents, refresh_type=refresh_type, similarity=similarity,
            timeout=timeout, return_embedding=return_embedding,
        )

        self.client = self._init_elastic_client(host=host, port=port, username=username, password=password,
                                           api_key=api_key, api_key_id=api_key_id, aws4auth=aws4auth, scheme=scheme,
                                           ca_certs=ca_certs, verify_certs=verify_certs,timeout=timeout)

        # configure mappings to ES fields that will be used for querying / displaying results
        if type(search_fields) == str:
            search_fields = [search_fields]

        #TODO we should implement a more flexible interal mapping here that simplifies the usage of additional,
        # custom fields (e.g. meta data you want to return)
        self.search_fields = search_fields
        self.text_field = text_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.embedding_dim = embedding_dim
        self.excluded_meta_data = excluded_meta_data
        self.faq_question_field = faq_question_field
        self.analyzer = analyzer
        self.return_embedding = return_embedding

        self.custom_mapping = custom_mapping
        self.index: str = index
        self.label_index: str = label_index
        if similarity in ["cosine", "dot_product"]:
            self.similarity = similarity
        else:
            raise Exception("Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between 'cosine' and 'dot_product'")
        if create_index:
            self._create_document_index(index)
            self._create_label_index(label_index)

        self.duplicate_documents = duplicate_documents
        self.refresh_type = refresh_type

    def _init_elastic_client(self,
                             host: Union[str, List[str]],
                             port: Union[int, List[int]],
                             username: str,
                             password: str,
                             api_key_id: Optional[str],
                             api_key: Optional[str],
                             aws4auth,
                             scheme: str,
                             ca_certs: Optional[str],
                             verify_certs: bool,
                             timeout: int) -> Elasticsearch:
        # Create list of host(s) + port(s) to allow direct client connections to multiple elasticsearch nodes
        if isinstance(host, list):
            if isinstance(port, list):
                if not len(port) == len(host):
                    raise ValueError("Length of list `host` must match length of list `port`")
                hosts = [{"host":h, "port":p} for h, p in zip(host,port)]
            else:
                hosts = [{"host": h, "port": port} for h in host]
        else:
            hosts = [{"host": host, "port": port}]

        if (api_key or api_key_id) and not(api_key and api_key_id):
            raise ValueError("You must provide either both or none of `api_key_id` and `api_key`")

        if api_key:
            # api key authentication
            client = Elasticsearch(hosts=hosts, api_key=(api_key_id, api_key),
                                        scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs, timeout=timeout)
        elif aws4auth:
            # aws elasticsearch with IAM
            # see https://elasticsearch-py.readthedocs.io/en/v7.12.0/index.html?highlight=http_auth#running-on-aws-with-iam
            client = Elasticsearch(
                hosts=hosts, http_auth=aws4auth, connection_class=RequestsHttpConnection, use_ssl=True, verify_certs=True, timeout=timeout)
        else:
            # standard http_auth
            client = Elasticsearch(hosts=hosts, http_auth=(username, password),
                                        scheme=scheme, ca_certs=ca_certs, verify_certs=verify_certs,
                                        timeout=timeout)

        # Test connection
        try:
            # ping uses a HEAD request on the root URI. In some cases, the user might not have permissions for that,
            # resulting in a HTTP Forbidden 403 response.
            if username in ["", "elastic"]:
                status = client.ping()
                if not status:
                    raise ConnectionError(
                        f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance "
                        f"at `{hosts}` and that it has finished the initial ramp up (can take > 30s)."
                    )
        except Exception:
            raise ConnectionError(
                f"Initial connection to Elasticsearch failed. Make sure you run an Elasticsearch instance at `{hosts}` and that it has finished the initial ramp up (can take > 30s).")
        return client

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents. In case if an index with the name already exists, it ensures that
        the embedding_field is present.
        """
        # check if the existing index has the embedding field; if not create it
        if self.client.indices.exists(index=index_name):
            if self.embedding_field:
                mapping = self.client.indices.get(index_name)[index_name]["mappings"]
                if self.embedding_field in mapping["properties"] and mapping["properties"][self.embedding_field]["type"] != "dense_vector":
                    raise Exception(f"The '{index_name}' index in Elasticsearch already has a field called '{self.embedding_field}'"
                                    f" with the type '{mapping['properties'][self.embedding_field]['type']}'. Please update the "
                                    f"document_store to use a different name for the embedding_field parameter.")
                mapping["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}
                self.client.indices.put_mapping(index=index_name, body=mapping)
            return

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.text_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.embedding_field:
                mapping["mappings"]["properties"][self.embedding_field] = {"type": "dense_vector", "dims": self.embedding_dim}

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _create_label_index(self, index_name: str):
        if self.client.indices.exists(index=index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "is_correct_answer": {"type": "boolean"},
                    "is_correct_document": {"type": "boolean"},
                    "origin": {"type": "keyword"},  # e.g. user-feedback or gold-label
                    "document_id": {"type": "keyword"},
                    "offset_start_in_doc": {"type": "long"},
                    "no_answer": {"type": "boolean"},
                    "model_id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                    "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                    #TODO add pipeline_hash and pipeline_name once we migrated the REST API to pipelines
                }
            }
        }
        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    # TODO: Add flexibility to define other non-meta and meta fields expected by the Document class
    def _create_document_field_map(self) -> Dict:
        return {
            self.text_field: "text",
            self.embedding_field: "embedding",
            self.faq_question_field if self.faq_question_field else "question": "question"
        }

    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        """Fetch a document by specifying its text id string"""
        index = index or self.index
        documents = self.get_documents_by_id([id], index=index)
        if documents:
            return documents[0]
        else:
            return None

    def get_documents_by_id(self, ids: List[str], index: Optional[str] = None) -> List[Document]:  # type: ignore
        """Fetch documents by specifying a list of text id strings"""
        index = index or self.index
        query = {"query": {"ids": {"values": ids}}}
        result = self.client.search(index=index, body=query)["hits"]["hits"]
        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def get_metadata_values_by_key(
        self,
        key: str,
        query: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        index: Optional[str] = None,
    ) -> List[dict]:
        """
        Get values associated with a metadata key. The output is in the format:
            [{"value": "my-value-1", "count": 23}, {"value": "my-value-2", "count": 12}, ... ]
        :param key: the meta key name to get the values for.
        :param query: narrow down the scope to documents matching the query string.
        :param filters: narrow down the scope to documents that match the given filters.
        :param index: Elasticsearch index where the meta values should be searched. If not supplied,
                      self.index will be used.
        """
        body: dict = {"size": 0, "aggs": {"metadata_agg": {"terms": {"field": key}}}}
        if query:
            body["query"] = {
                "bool": {
                    "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields, }}]
                }
            }
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append({"terms": {key: values}})
            if not body.get("query"):
                body["query"] = {"bool": {}}
            body["query"]["bool"].update({"filter": filter_clause})
        result = self.client.search(body=body, index=index)
        buckets = result["aggregations"]["metadata_agg"]["buckets"]
        for bucket in buckets:
            bucket["count"] = bucket.pop("doc_count")
            bucket["value"] = bucket.pop("key")
        return buckets

    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None,
                        batch_size: int = 10_000,duplicate_documents: Optional[str] = None):
        """
        Indexes documents for later queries in Elasticsearch.
        Behaviour if a document with the same ID already exists in ElasticSearch:
        a) (Default) Throw Elastic's standard error message for duplicate IDs.
        b) If `self.update_existing_documents=True` for DocumentStore: Overwrite existing documents.
        (This is only relevant if you pass your own ID when initializing a `Document`.
        If don't set custom IDs for your Documents or just pass a list of dictionaries here,
        they will automatically get UUIDs assigned. See the `Document` class for details)
        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
                          Advanced: If you are using your own Elasticsearch mapping, the key names in the dictionary
                          should be changed to what you have set for self.text_field and self.name_field.
        :param index: Elasticsearch index where the documents should be indexed. If not supplied, self.index will be used.
        :param batch_size: Number of documents that are passed to Elasticsearch's bulk function at a time.
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """

        if index and not self.client.indices.exists(index=index):
            self._create_document_index(index)

        if index is None:
            index = self.index
        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert duplicate_documents in self.duplicate_documents_options, \
            f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(document_objects, duplicate_documents)
        documents_to_index = []
        for doc in document_objects:
            _doc = {
                "_op_type": "index" if duplicate_documents == 'overwrite' else "create",
                "_index": index,
                **doc.to_dict(field_map=self._create_document_field_map())
            }  # type: Dict[str, Any]

            # cast embedding type as ES cannot deal with np.array
            if _doc[self.embedding_field] is not None:
                if type(_doc[self.embedding_field]) == np.ndarray:
                    _doc[self.embedding_field] = _doc[self.embedding_field].tolist()

            # rename id for elastic
            _doc["_id"] = str(_doc.pop("id"))

            # don't index query score and empty fields
            _ = _doc.pop("score", None)
            _ = _doc.pop("probability", None)
            _doc = {k:v for k,v in _doc.items() if v is not None}

            # In order to have a flat structure in elastic + similar behaviour to the other DocumentStores,
            # we "unnest" all value within "meta"
            if "meta" in _doc.keys():
                for k, v in _doc["meta"].items():
                    _doc[k] = v
                _doc.pop("meta")
            documents_to_index.append(_doc)

            # Pass batch_size number of documents to bulk
            if len(documents_to_index) % batch_size == 0:
                bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)
                documents_to_index = []

        if documents_to_index:
            bulk(self.client, documents_to_index, request_timeout=300, refresh=self.refresh_type)

    def write_labels(
        self, labels: Union[List[Label], List[dict]], index: Optional[str] = None, batch_size: int = 10_000
    ):
        """Write annotation labels into document store.
        :param labels: A list of Python dictionaries or a list of Haystack Label objects.
        :param batch_size: Number of labels that are passed to Elasticsearch's bulk function at a time.
        """
        index = index or self.label_index
        if index and not self.client.indices.exists(index=index):
            self._create_label_index(index)

        labels_to_index = []
        for l in labels:
            # Make sure we comply to Label class format
            if isinstance(l, dict):
                label = Label.from_dict(l)
            else:
                label = l

            # create timestamps if not available yet
            if not label.created_at:
                label.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
            if not label.updated_at:
                label.updated_at = label.created_at

            _label = {
                "_op_type": "index" if self.duplicate_documents == "overwrite" else "create",
                "_index": index,
                **label.to_dict()
            }  # type: Dict[str, Any]

            # rename id for elastic
            if label.id is not None:
                _label["_id"] = str(_label.pop("id"))

            labels_to_index.append(_label)

            # Pass batch_size number of labels to bulk
            if len(labels_to_index) % batch_size == 0:
                bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)
                labels_to_index = []

        if labels_to_index:
            bulk(self.client, labels_to_index, request_timeout=300, refresh=self.refresh_type)

    def update_document_meta(self, id: str, meta: Dict[str, str]):
        """
        Update the metadata dictionary of a document by specifying its string id
        """
        body = {"doc": meta}
        self.client.update(index=self.index, id=id, body=body, refresh=self.refresh_type)

    def get_document_count(self, filters: Optional[Dict[str, List[str]]] = None, index: Optional[str] = None,
                           only_documents_without_embedding: bool = False) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": self.embedding_field}}]

        if filters:
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_label_count(self, index: Optional[str] = None) -> int:
        """
        Return the number of labels in the document store
        """
        return self.get_document_count(index=index)

    def get_embedding_count(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> int:
        """
        Return the count of embeddings in the document store.
        """

        index = index or self.index

        body: dict = {"query": {"bool": {"must": [{"exists": {"field": self.embedding_field}}]}}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> List[Document]:
        """
        Get documents from the document store.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        result = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=return_embedding, batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        result = self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size)
        for hit in result:
            document = self._convert_es_hit_to_document(hit, return_embedding=return_embedding)
            yield document

    def get_all_labels(
        self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None, batch_size: int = 10_000
    ) -> List[Label]:
        """
        Return all labels in the document store
        """
        index = index or self.label_index
        result = list(self._get_all_documents_in_index(index=index, filters=filters, batch_size=batch_size))
        labels = [Label.from_dict(hit["_source"]) for hit in result]
        return labels

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Dict[str, List[str]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store
        """
        body: dict = {"query": {"bool": {}}}

        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": self.embedding_field}}]

        result = scan(self.client, query=body, index=index, size=batch_size, scroll="1d")
        yield from result

    def query(
        self,
        query: Optional[str],
        filters: Optional[Dict[str, List[str]]] = None,
        top_k: int = 10,
        custom_query: Optional[str] = None,
        index: Optional[str] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query as defined by the BM25 algorithm.
        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """

        if index is None:
            index = self.index

        # Naive retrieval without BM25, only filtering
        if query is None:
            body = {"query":
                        {"bool": {"must":
                                      {"match_all": {}}}}}  # type: Dict[str, Any]
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        # Retrieval via custom query
        elif custom_query:  # substitute placeholder for query and filters for the custom_query template string
            template = Template(custom_query)
            # replace all "${query}" placeholder(s) with query
            substitutions = {"query": f'"{query}"'}
            # For each filter we got passed, we'll try to find & replace the corresponding placeholder in the template
            # Example: filters={"years":[2018]} => replaces {$years} in custom_query with '[2018]'
            if filters:
                for key, values in filters.items():
                    values_str = json.dumps(values)
                    substitutions[key] = values_str
            custom_query_json = template.substitute(**substitutions)
            body = json.loads(custom_query_json)
            # add top_k
            body["size"] = str(top_k)

        # Default Retrieval via BM25 using the user query on `self.search_fields`
        else:
            body = {
                "size": str(top_k),
                "query": {
                    "bool": {
                        "should": [{"multi_match": {"query": query, "type": "most_fields", "fields": self.search_fields}}]
                    }
                },
            }

            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause

        if self.excluded_meta_data:
            body["_source"] = {"excludes": self.excluded_meta_data}

        logger.debug(f"Retriever query: {body}")
        result = self.client.search(index=index, body=body)["hits"]["hits"]

        documents = [self._convert_es_hit_to_document(hit, return_embedding=self.return_embedding) for hit in result]
        return documents

    def query_by_embedding(self,
                           query_emb: np.ndarray,
                           filters: Optional[Dict[str, List[str]]] = None,
                           top_k: int = 10,
                           index: Optional[str] = None,
                           return_embedding: Optional[bool] = None) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.
        :param query_emb: Embedding of the query (e.g. gathered from DPR)
        :param filters: Optional filters to narrow down the search space.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param top_k: How many documents to return
        :param index: Index name for storing the docs and metadata
        :param return_embedding: To return document embedding
        :return:
        """
        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not self.embedding_field:
            raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
        else:
            # +1 in similarity to avoid negative numbers (for cosine sim)
            body = {
                "size": top_k,
                "query": self._get_vector_similarity_query(query_emb, top_k)
            }
            if filters:
                filter_clause = []
                for key, values in filters.items():
                    if type(values) != list:
                        raise ValueError(f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                                         'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["script_score"]["query"] = {"bool": {"filter": filter_clause}}

            excluded_meta_data: Optional[list] = None

            if self.excluded_meta_data:
                excluded_meta_data = deepcopy(self.excluded_meta_data)

                if return_embedding is True and self.embedding_field in excluded_meta_data:
                    excluded_meta_data.remove(self.embedding_field)
                elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                    excluded_meta_data.append(self.embedding_field)
            elif return_embedding is False:
                excluded_meta_data = [self.embedding_field]

            if excluded_meta_data:
                body["_source"] = {"excludes": excluded_meta_data}

            logger.debug(f"Retriever query: {body}")
            result = self.client.search(index=index, body=body, request_timeout=300)["hits"]["hits"]

            documents = [
                self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, return_embedding=return_embedding)
                for hit in result
            ]
            return documents

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        if self.similarity == "cosine":
            similarity_fn_name = "cosineSimilarity"
        elif self.similarity == "dot_product":
            similarity_fn_name = "dotProduct"
        else:
            raise Exception("Invalid value for similarity in ElasticSearchDocumentStore constructor. Choose between \'cosine\' and \'dot_product\'")

        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    # offset score to ensure a positive range as required by Elasticsearch
                    "source": f"{similarity_fn_name}(params.query_vector,'{self.embedding_field}') + 1000",
                    "params": {"query_vector": query_emb.tolist()},
                },
            }
        }
        return query

    def _convert_es_hit_to_document(
            self,
            hit: dict,
            return_embedding: bool,
            adapt_score_for_embedding: bool = False,

    ) -> Document:
        # We put all additional data of the doc into meta_data and return it in the API
        meta_data = {k:v for k,v in hit["_source"].items() if k not in (self.text_field, self.faq_question_field, self.embedding_field)}
        name = meta_data.pop(self.name_field, None)
        if name:
            meta_data["name"] = name

        score = hit["_score"] if hit["_score"] else None
        if score:
            if adapt_score_for_embedding:
                score = self._scale_embedding_score(score)
                if self.similarity == "cosine":
                    probability = (score + 1) / 2  # scaling probability from cosine similarity
                elif self.similarity == "dot_product":
                    probability = float(expit(np.asarray(score / 100)))  # scaling probability from dot product
            else:
                probability = float(expit(np.asarray(score / 8)))  # scaling probability from TFIDF/BM25
        else:
            probability = None

        embedding = None
        if return_embedding:
            embedding_list = hit["_source"].get(self.embedding_field)
            if embedding_list:
                embedding = np.asarray(embedding_list, dtype=np.float32)

        document = Document(
            id=hit["_id"],
            text=hit["_source"].get(self.text_field),
            meta=meta_data,
            score=score,
            probability=probability,
            question=hit["_source"].get(self.faq_question_field),
            embedding=embedding,
        )
        return document

    def _scale_embedding_score(self, score):
        return score - 1000

    def describe_documents(self, index=None):
        """
        Return a summary of the documents in the document store
        """
        if index is None:
            index = self.index
        docs = self.get_all_documents(index)

        l = [len(d.text) for d in docs]
        stats = {"count": len(docs),
                 "chars_mean": np.mean(l),
                 "chars_max": max(l),
                 "chars_min": min(l),
                 "chars_median": np.median(l),
                 }
        return stats

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).
        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index)
            logger.info(f"Updating embeddings for all {document_count} docs ...")
        else:
            document_count = self.get_document_count(index=index, filters=filters,
                                                     only_documents_without_embedding=True)
            logger.info(f"Updating embeddings for {document_count} docs without embeddings ...")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings
        )

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit, return_embedding=False) for hit in result_batch]
                embeddings = retriever.embed_passages(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                       f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                       "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()")
                doc_updates = []
                for doc, emb in zip(document_batch, embeddings):
                    update = {"_op_type": "update",
                              "_index": index,
                              "_id": doc.id,
                              "doc": {self.embedding_field: emb.tolist()},
                              }
                    doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type)
                progress_bar.update(batch_size)

    def delete_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.
        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        logger.warning(
                """DEPRECATION WARNINGS: 
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, filters)

    def delete_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None):
        """
        Delete documents in an index. All documents are deleted if no filters are passed.
        :param index: Index name to delete the document from.
        :param filters: Optional filters to narrow down the documents to be deleted.
        :return: None
        """
        index = index or self.index
        query: Dict[str, Any] = {"query": {}}
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                )
                query["query"]["bool"] = {"filter": filter_clause}
        else:
            query["query"] = {"match_all": {}}
        self.client.delete_by_query(index=index, body=query, ignore=[404])
        # We want to be sure that all docs are deleted before continuing (delete_by_query doesn't support wait_for)
        if self.refresh_type == "wait_for":
            time.sleep(2)

class OpenDistroElasticsearchDocumentStore(ElasticsearchDocumentStore):
    """
    Document Store using the Open Distro for Elasticsearch. It is compatible with the AWS Elasticsearch Service.
    In addition to native Elasticsearch query & filtering, it provides efficient vector similarity search using
    the KNN plugin that can scale to a large number of documents.
    """

    def _create_document_index(self, index_name: str):
        """
        Create a new index for storing documents.
        """

        if self.custom_mapping:
            mapping = self.custom_mapping
        else:
            mapping = {
                "mappings": {
                    "properties": {
                        self.name_field: {"type": "keyword"},
                        self.text_field: {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "path_match": "*",
                                "match_mapping_type": "string",
                                "mapping": {"type": "keyword"}}}
                    ],
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": self.analyzer,
                            }
                        }
                    }
                }
            }
            if self.embedding_field:
                if self.similarity == "cosine":
                    similarity_space_type = "cosinesimil"
                elif self.similarity == "dot_product":
                    similarity_space_type = "l2"
                else:
                    raise Exception(
                        f"Similarity function {self.similarity} is not supported by OpenDistroElasticsearchDocumentStore."
                    )
                mapping["settings"]["knn"] = True
                mapping["settings"]["knn.space_type"] = similarity_space_type
                mapping["mappings"]["properties"][self.embedding_field] = {
                    "type": "knn_vector",
                    "dimension": self.embedding_dim,
                }

        try:
            self.client.indices.create(index=index_name, body=mapping)
        except RequestError as e:
            # With multiple workers we need to avoid race conditions, where:
            # - there's no index in the beginning
            # - both want to create one
            # - one fails as the other one already created it
            if not self.client.indices.exists(index=index_name):
                raise e

    def _get_vector_similarity_query(self, query_emb: np.ndarray, top_k: int):
        """
        Generate Elasticsearch query for vector similarity.
        """
        query = {"knn": {self.embedding_field: {"vector": query_emb.tolist(), "k": top_k}}}
        return query

    def _scale_embedding_score(self, score):
        return score