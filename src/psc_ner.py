import spacy
import re

def ner(prompt):
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(prompt)

    # Find quoted phrases and their character spans
    quoted_matches = list(re.finditer(r"'(.*?)'|\"(.*?)\"", prompt))
    quoted_phrases = [m.group(1) if m.group(1) else m.group(2) for m in quoted_matches]
    quoted_spans = [(m.start(), m.end()) for m in quoted_matches]

    # Extract PERSON names and find their token spans
    person_name_list = [ ent.text for ent in doc.ents if ent.label_ == "PERSON" ]
    person_spans = [(ent.start, ent.end) for ent in doc.ents if ent.label_ == "PERSON"]

    # Helper to check if a token span overlaps a quoted phrase
    def overlaps_quote(span):
        for start, end in quoted_spans:
            if not (doc[span.start].idx + len(doc[span.start:span.end].text) <= start or doc[span.start].idx >= end):
                return True
        return False

    def overlaps_person(chunk, person_spans):
        for ps_start, ps_end in person_spans:
            if chunk.start < ps_end and chunk.end > ps_start:
                return True
        return False

    # Extract noun chunks, excluding PERSONs and quoted phrases
    potential_indirect_pcc_list = []
    for chunk in doc.noun_chunks:
        # Skip PERSON
        if overlaps_person(chunk, person_spans):
            continue
        # Skip if overlaps a quoted phrase
        if overlaps_quote(chunk):
            continue
        potential_indirect_pcc_list.append(chunk.text)

    return {
        "person_name_list": person_name_list,
        "potential_indirect_pcc_list": potential_indirect_pcc_list,
        "potential_direct_pcc_list": quoted_phrases
    }