from dataclasses import dataclass
import bs4
from bs4 import BeautifulSoup
import lxml
from os.path import basename, splitext
import json

@dataclass
class Document:
    text: str
    meta: str


@dataclass
class Author:
    firstname: str
    middlename: str
    surname: str
    org_name_department: str
    org_name_institution: str


class TEIFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.soup = read_tei(filename)
        self._text = None
        self._abstract = ""
        self._title = ""
        
    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def authors(self):
        result = []
        for author in self.soup.analytic.find_all('author'):
            persname = author.persname
            if not persname:
                continue
            firstname = elem_to_text(persname.find("forename", type="first"))
            middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            department = elem_to_text(persname.find("orgName", type="department"))
            institution = elem_to_text(persname.find("orgName", type="institution"))
            author = {
                "firstname": firstname,
                "middlename": middlename,
                "surname": surname,
                "ord_department": department,
                "org_institution": institution
            }
            result.append(author)
        return result

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def text(self):
        if not self._text:
            divs_text = []
            for div in self.soup.body.find_all("div"):
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    for child in div.children:
                        if(not child.name == "listbibl"):
                            if(isinstance(child, bs4.element.NavigableString)):
                                divs_text.append(str(child))
                            elif(isinstance(child, bs4.element.Tag)):
                                divs_text.append(child.text)
            self._text = divs_text
        return self._text
    
    def plain_text(self):
        # For debugging only
        return " ".join(self.text)
    
    def metadata_dict(self):
        return {
            "title": self.title,
            "authors": self.authors,
            "DOI": self.doi
        }
    
    def to_dict(self):
        return {
            "text": "\n".join(self.text),
            "meta": self.metadata_dict()
        }


# Helper
def read_tei(tei_file):
    with open(tei_file, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')
        return soup
    raise RuntimeError('Cannot generate a soup from the input')

def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default

def basename_without_ext(path):
    base_name = basename(path)
    stem, ext = splitext(base_name)
    if stem.endswith('.tei'):
        # Return base name without tei file
        return stem[0:-4]
    else:
        return stem

def tei_to_csv_entry(tei_file):
    tei = TEIFile(tei_file)
    base_name = basename_without_ext(tei_file)
    return base_name, tei.title, tei.authors, tei.doi, tei.text