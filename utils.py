import re
from langchain_core.documents import Document
from typing import Dict, List, Any


def format_response(llm_response: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    This function takes the LLM response as a dictionary and formats it.
    It returns a Dictionary containing:
        - Model Answer
        - Sources used to create the answer
    """
    answer = llm_response['answer']
    sources = format_sources(llm_response['source_documents'])

    response = {'answer': answer, "sources": sources}
    return response


def clean_source_filename(tmp_filename_path: str) -> str:
    """
    This function cleans the temporal filename to just show the original file name. 
    Since we are storing the loaded files as temporary ones, a temporary one is created.
    """
    tmp_filename = tmp_filename_path.split('/')[2]
    clean_filename = re.search(r'^(.+?\.pdf)', tmp_filename)
    return clean_filename.group()


def format_sources(list_sources: List[Document]) -> List[str]:
    """
    This function takes a list of Documents and it format them to return as
    a list of strings
    """
    grouped_sources = {}
    sources = []

    for document in list_sources:
        source = clean_source_filename(document.metadata['source'])
        page = document.metadata['page']
        
        if source in grouped_sources:
            grouped_sources[source].append(page)
        else:
            grouped_sources[source] = [page]

    for key in grouped_sources.keys():
        pages = [str(pag) for pag in grouped_sources[key]]
        pages_text = "Pags: " + ", ".join(map(str, sorted(set(pages))))
        sources.append(f"{key} ({pages_text})")

    return sources