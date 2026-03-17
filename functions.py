
# functions needed for citation classification of pdfs

import fitz
import re
import unicodedata

# extract full text from pdf

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
       text+=page.get_text()
    return text 


# split text into body and references, using regex to find the "References" or "Bibliography" section, and also to detect appendices that may follow references
# The function returns the body and references as separate strings. If no references section is found, it returns the entire text as the body and an empty string for references. If an appendix is detected after the references, it is included in the body to ensure that only the actual references are separated out.

def split_body_and_references(text):
    # Locate References / Bibliography header
    ref_match = re.search(
        r"(?im)^\s*(references|bibliography)\s*$",
        text
    )

    if not ref_match:
        return text, ""

    body = text[:ref_match.start()]
    refs_and_after = text[ref_match.end():]

    # Appendix detection:
    # 1) APPENDIX A:
    # 2) A. on its own line, followed by a non-citation line

    appendix_match = re.search(
        r"""(?im)^(
            \s*appendix\s+[A-Z]\s*:?
            |
            \s*[A-Z]\.\s*$\n(?!\s*[A-Z][a-z]+,\s+[A-Z])
        )""",
        refs_and_after,
        re.VERBOSE
    )

    if appendix_match:
        refs = refs_and_after[:appendix_match.start()]
        appendix = refs_and_after[appendix_match.start():]
        body = body.rstrip() + "\n\n" + appendix
    else:
        refs = refs_and_after

    return body.strip(), refs.strip()


# Build a mapping from numeric reference numbers to their corresponding reference entries, based on the provided reference text and target reference patterns. 
# The function takes in the reference text (as a string) and a dictionary of target references, where each key is a reference identifier and the value is a list of regex patterns to match against the reference entries. 
# It returns a dictionary mapping numeric reference numbers (as strings) to their corresponding reference keys and entries.

def build_numeric_reference_map(ref_text, target_refs):
    """
    Returns:
    {
      "12": {
        "ref_key": "...",
        "reference_entry": "A. Bailer-Jones et al., ..."
      }
    }
    """
    ref_map = {}

    for line in ref_text.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+(.*)", line)
        if not m:
            continue

        number, entry = m.groups()

        for ref_key, patterns in target_refs.items():
            for pattern in patterns:
                if re.search(pattern, entry, flags=re.IGNORECASE):
                    ref_map[number] = {
                        "ref_key": ref_key,
                        "reference_entry": entry.strip()
                    }
                    break

    return ref_map





# Extracts a snippet of text from a paragraph that is anchored around a specific match (e.g., a citation). 
# The function takes in the paragraph, the start and end indices of the match, and an optional character window size (default is 300 characters).

def extract_anchored_context(paragraph, match_start, match_end, char_window=400):
    start = max(0, match_start - char_window)
    end = min(len(paragraph), match_end + char_window)
    snippet = paragraph[start:end]

    snippet = re.sub(r'^[^.!?]*[.!?]', '', snippet)
    snippet = re.sub(r'[.!?][^.!?]*$', '', snippet)

    return snippet.strip()

# Extracts contexts around citations in the body text. 
# It identifies paragraphs, filters out short ones, and then searches for matches based on author patterns and numeric reference numbers. 
# For each match found, it extracts a context snippet using the anchored context function.

def extract_citation_contexts(body_text, author_patterns, numeric_numbers):
    paragraphs = re.split(r'\n\s*\n', body_text)
    contexts = []

    for paragraph in paragraphs:
        if len(paragraph.strip()) < 80:
            continue

        for pattern in author_patterns:
            for m in re.finditer(pattern, paragraph, flags=re.IGNORECASE):
                contexts.append(extract_anchored_context(paragraph, m.start(), m.end()))

        for num in numeric_numbers:
            for m in re.finditer(rf"\[\s*{num}(\s*,\s*\d+)*\s*\]", paragraph):
                contexts.append(extract_anchored_context(paragraph, m.start(), m.end()))

    return contexts

# function to clean and normalize extracted text from PDFs, addressing common issues such as non-breaking spaces, soft hyphens, and various dash types.

def clean_pdf_text(text: str) -> str:
    # Normalize Unicode (very important for PDFs)
    text = unicodedata.normalize("NFKC", text)

    # Replace non-breaking spaces with normal spaces
    text = text.replace("\xa0", " ")

    # Remove soft hyphen (often invisible but breaks matching)
    text = text.replace("\u00AD", "")

    # Normalize different dash types to standard hyphen
    text = re.sub(r"[‐-–—−]", "-", text)

    # Fix hyphenated line breaks: "Bailer-\nJones" → "Bailer-Jones"
    text = re.sub(r"-\s*\n\s*", "-", text)

    # Replace remaining line breaks with space
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()