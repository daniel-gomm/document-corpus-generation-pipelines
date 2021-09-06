from .imrad_classification import ClassificationHandler
import abc
import re
import json
import nltk
import logging
import numpy as np
import copy

from pandas import DataFrame
from typing import List, Dict
from haystack.preprocessor import PreProcessor
from ..database.rocksDB import RocksDBAdapter
from .DocumentFields import MetadataFields

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')

from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline
import torch



#Helpers

def _string_alpha_percentage(s:str):
    """Calculate the percentage of non alpha numeric words in a given string

        Args:
            s (str): String that is looked at for the calculation.
        """
    words = word_tokenize(s)
    num_non = 0
    for word in words:
        if not word.isalpha() or len(word) < 2:
            num_non += 1
    return num_non/(len(words)*1.0)

def _string_pattern_percentage(s:str, filter):
    """Calculate the percentage of a given expression in a given string

        Args:
            s (str): String that is looked at for the calculation.
            filter: Expression thats percentage is calculated.
        """
    words = len(word_tokenize(s))
    matches = len(re.findall(filter, s))
    return matches/(words*1.0)


class Processor(metaclass=abc.ABCMeta):
    # Interface for processing steps
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process') and callable(subclass.process) or NotImplemented)

    @abc.abstractmethod
    def process(self, documents: List[Dict]) -> List[Dict]:
        """Processes a batch of documents, modifies them and outputs the result.

        Args:
            documents (List[Dict]): Input documents which should have the structure {'text':..., 'meta':{...}}.

        Raises:
            NotImplementedError: Raised if the Method is not implemented by subclass.

        Returns:
            List[Dict]: Output documents modified by the processor.
        """
        raise NotImplementedError

class IndexedProcessor(metaclass=abc.ABCMeta):
    # Interface for processing steps
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process_index') and callable(subclass.process_index) or NotImplemented)

    @abc.abstractmethod
    def process_index(self, documents: List[Dict], index: int) -> List[Dict]:
        """Processes a batch of documents, modifies them and outputs the result. Adds an index field to the process function.

        Args:
            documents (List[Dict]): Input documents which should have the structure {'text':..., 'meta':{...}}.
            index (int): Integer value to be used in dividing the processing load along external resources, e.g. GPUs.

        Raises:
            NotImplementedError: Raised if the Method is not implemented by subclass.

        Returns:
            List[Dict]: Output documents modified by the processor.
        """
        raise NotImplementedError

# General Processors


class HaystackPreProcessor(Processor):

    def __init__(self, preprocessor: PreProcessor):
        """Wrapper for Haystack PreProcessor.

        Args:
            preprocessor (PreProcessor): Haystack PreProcessor that should be applied.
        """
        self._preprocessor = preprocessor

    def process(self, documents: List[Dict]) -> List[Dict]:
        docs = []
        for document in documents:
            docs.extend(self._preprocessor.process(document))
        return docs


class SplitByLinePreProcessor(Processor):

    def __init__(self, split_len: int = 100, minimal_split_len: int = 50):
        """Alternative Preprocessor that splits documents respecting boundaries set be linebreaks.

        Args:
            split_len (int, optional): Maximum length of a segment. Defaults to 100.
            minimal_split_len (int, optional): Minimal length of a segment. Defaults to 50.
        """
        self._split_len = split_len
        self._minimal_split_len = minimal_split_len

    def process(self, documents: List[Dict]) -> List[Dict]:
        #Construct segments by appending as many sentences as possible from the same line without exceeding the upper limit. If there are not enough sentences to satisfy the lower limit paragraphs from the next section are appended.
        docs = []
        for document in documents:
            lines = document["text"].split("\n")
            meta = document["meta"]
            split_id = 0
            p_cleaned = []
            for line in lines:
                p_tokenized = word_tokenize(line)
                if len(p_tokenized) < 6:
                    continue
                elif self._minimal_split_len < len(p_tokenized) < self._split_len:
                    p_cleaned.append(line)
                else:
                    p_sentences = sent_tokenize(line)
                    current_sent = ""
                    len_current = 0
                    for sent in p_sentences:
                        words = word_tokenize(sent)
                        len_sent = len(words)
                        if len_current >= self._split_len:
                            while len_sent >= self._split_len:
                                w = words[0:np.minimum(
                                    self._split_len, len(words))]
                                p_cleaned.append(
                                    TreebankWordDetokenizer().detokenize(w))
                                words = words[self._split_len:]
                                len_sent = len(words)
                        elif len_current + len_sent < self._split_len:
                            current_sent += " " + sent
                            len_current += len_sent
                        else:
                            p_cleaned.append(current_sent)
                            current_sent = sent
                            len_current = len_sent
            for paragraph in p_cleaned:
                document = {}
                document["meta"] = copy.deepcopy(meta)
                document["meta"]["_split_id"] = split_id
                split_id += 1
                document["text"] = paragraph
                docs.append(document)
        return docs

# Metadata Processors


class MetadataFieldDiscarder(Processor):

    def __init__(self, fields_to_discard: List):
        """Discards provided metadata fields from meta dictionary.

        Args:
            fields_to_discard (List): list of names of the fields that should be discarded.
        """
        self._fields_to_discard = fields_to_discard

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            for field in self._fields_to_discard:
                document["meta"].pop(field, None)
        return documents
    
class MetadataFieldAdder(Processor):

    def __init__(self, field_name:str, default_value):
        """Adds a field with a specified default value to each document.

        Args:
            field_name (str): Name of the field to be added.
            default_value ([type]): Default value that is assigned to the field.
        """
        self._field_name = field_name
        self._default_value = default_value

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            document["meta"][self._field_name] = self._default_value
        return documents


class MetadataArxiveEnricher(Processor):

    def __init__(self, id_field: str, db: RocksDBAdapter, discard_entries_without_metadata: bool = True):
        """Adds arXive metadata to the metadata dictionary.

        Args:
            id_field (str): Field in metadata that contains the arXive ID.
            db (RocksDBAdapter): Connector to RocksDB that contains the arXive Metadata.
            discard_entries_without_metadata (bool, optional): Flags if entries for which no metadata is found should be discarded. Defaults to True.
        """
        self._db = db
        self._id_field = id_field
        self._discard_missing = discard_entries_without_metadata

    def process(self, documents: List[Dict]) -> List[Dict]:
        ids = list(map(lambda doc: doc["meta"][self._id_field], documents))
        response = self._db.get_all(ids)
        metadata = json.loads(response.text)
        docs_to_return = []
        for document in documents:
            received_meta = metadata[document["meta"][self._id_field]]
            if(received_meta == "Data unavailable"):
                document["meta"].update(
                    {"unavailable metadata": "Metadata not found in database."})
                if not self._discard_missing:
                    docs_to_return.append(document)
            elif received_meta != "Data unavailable":
                document["meta"].update(json.loads(received_meta))
                docs_to_return.append(document)
        return docs_to_return


class MetadataMagArxiveLinker(Processor):

    def __init__(self, dataframe: DataFrame, column_to_match: str, column_to_add: str, field_to_match: str = "arixive-id"):
        """Adds the MAG-ID to documents based on their arXiv-ID.

        Args:
            dataframe (DataFrame): datframe containing pairs of MAG-IDs and arXive-IDs.
            column_to_match (str): Columnname containing the arXive-IDs.
            column_to_add (str): Columnname containing the MAG-IDs.
            field_to_match (str, optional): Metadata field that contains the arXive-ID. Defaults to "arixive-id".
        """
        self._column_to_match = column_to_match
        self._column_to_add = column_to_add
        self._dataframe = dataframe[[
            self._column_to_add, self._column_to_match]]
        self._field_to_match = field_to_match

    def process(self, documents: List[Dict]) -> List[Dict]:
        docs_to_return = []
        for document in documents:
            try:
                document["meta"][self._column_to_add] = self._find_match(
                    document["meta"][self._field_to_match])
                docs_to_return.append(document)
            except IndexError:
                logging.info("No matching id found for {}".format(
                    document["meta"][self._field_to_match]))
        return documents

    def _find_match(self, value):
        matching_value = self._dataframe[self._dataframe[self._column_to_match]
                                         == value][self._column_to_add].iloc[0]
        return int(matching_value)

class MetadataFlattener(Processor):

    def __init__(self, fields_to_flatten:List[str]):
        """Flattens the specified fiels by serialising them to json objects.

        Args:
            fields_to_flatten (List[str]): List of fields that should be flattened.
        """
        self._fields_to_flatten = fields_to_flatten
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        docs_to_return = []
        for document in documents:
            for key in [key for key in document['meta'].keys() if key in self._fields_to_flatten]:
                document["meta"][key] = json.dumps(document["meta"][key])
            docs_to_return.append(document)
        return docs_to_return

class MetadataReplaceFilter(Processor):

    def __init__(self, filters: List[str], replacement: str, field:str):
        """Replace a substring of the text.

        Args:
            filters (List[str]): Specifies the substrings that should be replaced.
            replacement (str): Replacement for instances of the filter.
            field (str): Metadata field the filter is applied to.
        """
        self._filters = filters
        self._replacement = replacement
        self._field = field

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            for filter in self._filters:
                document["metadata"][filter] = re.sub(
                    self._filter, self._replacement, document["metadata"][filter])
        return documents



# Text Processors


class TextKeywordCut(Processor):

    def __init__(self, keyword: str, cut_off_upper_part: bool = True):
        """Cuts off the text above or below a certain keyword.

        Args:
            keyword (str): Keyword on anchoring the cut.
            cut_off_upper_part (bool, optional): If True the part above the keyword is cut of, if False the part below the keyword is cut of. Defaults to True.
        """
        self._keyword = keyword.lower()
        self._cut__off_upper_part = cut_off_upper_part

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            text = document["text"]
            if self._keyword in text.lower():
                if self._cut__off_upper_part:
                    text_substring = text.lower().partition(self._keyword)[2]
                    text = text[-(len(text_substring)+len(self._keyword)):]
                else:
                    text_substring = text.lower().partition(self._keyword)[0]
                    text = text[0:(len(text_substring))]
                document["text"] = text
        return documents


class TextReplaceFilter(Processor):

    def __init__(self, filter: str, replacement: str):
        """Replace a substring of the text.

        Args:
            filter (str): Specifies the substring that should be replaced.
            replacement (str): Replacement for instances of the filter.
        """
        self._filter = filter
        self._replacement = replacement

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            document["text"] = re.sub(
                self._filter, self._replacement, document["text"])
        return documents


class TextAppendMetadataField(Processor):

    def __init__(self, field_to_attach: str, metdata_field_content_before_text: bool = True, line_mode:bool=False):
        """Appends a metadata field to the documents text field.

        Args:
            field_to_attach (str): Field which should be attached (e.g. 'abstract').
            metdata_field_content_before_text (bool, optional): True if field should be attached in front of existing text, False if it should be appended. Defaults to True.
        """
        self._field_to_attach = field_to_attach
        self._metdata_field_content_before_text = metdata_field_content_before_text
        if line_mode:
            self._connecting_token = "\n"
        else:
            self._connecting_token = " "

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            meta_field_content = document["meta"][self._field_to_attach].replace("\n", " ")
            if meta_field_content != None:
                if self._metdata_field_content_before_text:
                    document["text"] = meta_field_content + \
                        self._connecting_token + document["text"]
                else:
                    document["text"] += self._connecting_token + meta_field_content
        return documents

#Text Processors -> Sentence wise

class TextSentenceDiscardNonAlpha(Processor):

    def __init__(self, maximum_percentage:float=0.3):
        """Discards sentences from documents when lines contain more than a specified percentage of non alphanumeric words.

        Args:
            maximum_percentage (float, optional): Maximum percentage of non alphanumeric words allowed. Defaults to 0.3.
        """
        self._maximum_percentage = maximum_percentage
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            sentences = []
            for sentence in sent_tokenize(document['text']):
                if _string_alpha_percentage(sentence) <= self._maximum_percentage:
                    sentences.append(sentence)
            document['text'] = " ".join(sentences)
        return documents

class TextSentenceDiscardFilter(Processor):

    def __init__(self, filter:str, maximum_percentage:float=0.2):
        """Discards sentences from documents when lines contain more than a specified percentage of words that match a specified filter.

        Args:
            maximum_percentage (float, optional): Maximum percentage of non words that match the filter allowed. Defaults to 0.3.
        """
        self._filter = filter
        self._maximum_percentage = maximum_percentage
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            sentences = []
            for sentence in sent_tokenize(document['text']):
                if _string_alpha_percentage(sentence) <= self._maximum_percentage:
                    sentences.append(sentence)
            document['text'] = " ".join(sentences)
        return documents


# Filter Processors

class FilterOnMetadataValue(Processor):

    def __init__(self, metadata_field: str, values: List, discard_docs_wo_value: bool = True):
        """Filters out documents which don't have any of the provided values in a  specific field.

        Args:
            metadata_field (str): Field that's value is assessed.
            values (List): Possible values.
            discard_docs_wo_value (bool, optional): [description]. Defaults to True.
        """
        self._metadata_field = metadata_field
        self._values = values
        self._discard_wo_values = discard_docs_wo_value

    def process(self, documents: List[Dict]) -> List[Dict]:
        if self._discard_wo_values:
            return list(filter(lambda d: self._contains_value(d["meta"][self._metadata_field]), documents))
        else:
            return list(filter(lambda d: not self._contains_value(d["meta"][self._metadata_field]), documents))

    def _contains_value(self, text: str):
        return any(substring in text for substring in self._values)


class FilterExistingDocuments(Processor):

    def __init__(self, metadata_field: str, existing_ids: List[str]):
        """Filters documents based on a blacklist of strings that is not allowed to be contained in a specified metadata field.

        Args:
            metadata_field (str): Metadata field to compare to list.
            existing_ids (List[str]): List containing already registered ids.
        """
        self._metadata_field = metadata_field
        self._existing_ids = existing_ids

    def process(self, documents: List[Dict]) -> List[Dict]:
        return list(filter(lambda d: not d['meta'][self._metadata_field] in self._existing_ids, documents))


class FilterShortDocuments(Processor):

    def __init__(self, minimal_length:int):
        """Discards documents that are to short.

        Args:
            minimal_length (int): Minimal length at which documents are not discarded.
        """
        self._minimal_length = minimal_length
    
    def process(self, documents: List[Dict]) -> List[Dict]:
        return list(filter(lambda d: len(word_tokenize(d['text'])) >= self._minimal_length, documents))


# Line wise Processors

class LineDiscardProcessor(Processor):

    def __init__(self, percentage: float = 0.5) -> None:
        """Discards lines from documents when lines contain more than a specified percentage of non alphanumeric words.

        Args:
            percentage (float, optional): Maximum percentage of non alphanumeric words allowed. Defaults to 0.5.
        """
        self._percentage = percentage
        super().__init__()

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            lines = []
            for line in document["text"].split("\n"):
                if _string_alpha_percentage(line) <= self._percentage:
                    lines.append(line)
            document["text"] = "\n".join(lines)
        return documents

class LineUnarxiveGenerator(Processor):
    #Processor that converts an unarXive paper from its default structure to a document thats lines resemble paragraphs in the original paper.

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            paragraphs = document["text"].split("\n\n")
            p_cleaned = []
            for paragraph in paragraphs:
                paragraph = paragraph.strip("\n")
                lines = paragraph.split("\n")
                if lines:
                    if len(word_tokenize(lines[0])) < 5:
                        lines.remove(lines[0])
                    elif not lines[0].endswith("."):
                        lines.insert(0, lines[0] + ".")
                        lines.pop(1)
                p_cleaned.append(" ".join(lines))
            document['text'] = "\n".join(p_cleaned)
        return documents


# IMRaD


class IMRaDClassification(Processor):
    def __init__(self, model_path:str, model_identifier:str="allenai/scibert_scivocab_cased"):
        """Classifies sentences in the documents text field into Introduction, Related Work, Methodology, Results and Discussion.

        Args:
            model_path (str): Path of the model.
            model_identifier (str): Identifier to specify the used model. Defaults to 'allenai/scibert_scivocab_cased'.
        """
        global gpu_counter
        gpu_counter=0
        self._model = BertForSequenceClassification.from_pretrained(model_identifier, num_labels=5, output_attentions=False, output_hidden_states=False)
        self._model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

        if torch.cuda.is_available():
            self._classificationPipeline0 = TextClassificationPipeline(model=self._model, tokenizer=BertTokenizer.from_pretrained(model_identifier), device=0)
        if not (torch.cuda.is_available()):
            self._classificationPipeline0 = TextClassificationPipeline(model=self._model, tokenizer=BertTokenizer.from_pretrained(model_identifier), device=-1)
        self._model = BertForSequenceClassification.from_pretrained(model_identifier, num_labels=5, output_attentions=False, output_hidden_states=False)
        self._model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
        

    def process(self, documents: List[Dict]) -> List[Dict]:
        helperArray=[]
        for document in documents:
            helperArray.append(document["text"])
        changedLabelsCounter=0

        labelArray=self._classificationPipeline0(helperArray)
        
        for i in range (0, len(labelArray)):
            label=labelArray[i]["label"]
            if (label=="LABEL_0"):
                label="intro"
                
            if (label=="LABEL_1"):
                label="related"
                
            if (label=="LABEL_2"):
                label="method"
                
            if (label=="LABEL_3"):
                label="results"
                
            if (label=="LABEL_4"):
                label="discussion"
            documents[i]["meta"]["Section"]=label
            #documents[i]["meta"]["_status"]=1
        
        for i in range(0, len(documents)):
            if (i<(len(documents)+2)):
                if (documents[i]["meta"]["mag_id"]==documents[np.minimum(i+1, len(documents)-1)]["meta"]["mag_id"]==documents[np.minimum(i+2, len(documents)-1)]["meta"]["mag_id"]):
                    if ((documents[i]["meta"]["_split_id"]+2)==(documents[np.minimum(i+1, len(documents)-1)]["meta"]["_split_id"]+1)==(documents[np.minimum(i+2, len(documents)-1)]["meta"]["_split_id"])):
                        if(documents[i]["meta"]["Section"]==documents[i+2]["meta"]["Section"]):
                            if (documents[np.minimum(i+1, len(documents)-1)]["meta"]["Section"]!=documents[i]["meta"]["Section"]):
                                documents[np.minimum(i+1, len(documents)-1)]["meta"]["Section"]=documents[i]["meta"]["Section"]
                                changedLabelsCounter=changedLabelsCounter+1
            documents[i]["meta"]["_status"]=4
        
        return documents

# NE Processors

class StringMatchingProcessor(Processor):

    def __init__(self, entities: dict, info_field_name: str, info_key: str = 'info', flag_field_name:str=None):
        """Links entities with corresponding information.

        Args:
            entities (dict): Dictionary of entities in the format: {'entityName':'entityInformation'}.
            info_field_name (str): Name of the metadata field added.
            info_key (str): Name of the key holding the entity information
            flag_field_name: Name of the field that flags whether an entity has been found. Defaults to None.
        """
        if flag_field_name:
            self._flag_field_name = flag_field_name
        else:
            self._flag_field_name = "has_"+info_field_name
        self._entities = entities
        self._field_to_add = info_field_name
        self._info_key = info_key

    def process(self, documents: List[Dict]) -> List[Dict]:
        for document in documents:
            if not self._field_to_add in document['meta']:
                document['meta'][self._field_to_add] = []
            document['meta'][self._flag_field_name] = 'false'
            for entity_key, entity_value in self._entities.items():
                regex = re.compile(r'\b' + re.escape(entity_key) + r'\b')
                found_entities = regex.finditer(document['text'].lower())
                for found_entity in found_entities:
                    document['meta'][self._flag_field_name] = 'true'
                    document['meta'][self._field_to_add].append(
                        {'title': entity_key, self._info_key: entity_value, 'span': found_entity.span()})
        return documents
