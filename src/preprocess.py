from collections import Counter

entity_name = [
    'Organization',
    'Observatory',
    'CelestialObject',
    'Event',
    'CelestialRegion',
    'Identifier'
]

ner_tags = ["B-"+entity for entity in entity_name] + ["I-"+entity for entity in entity_name]

tag_to_id = {"O" : len(ner_tags)}
for i in range(len(ner_tags)):
    tag_to_id[ner_tags[i]] = i

def process_entity_tag(data, ner_tags=ner_tags, sample=None):
    """
    Process ner tags based on selected entities
 
    Args:
        data: (hugging face dateset).
        ner_tags (list): ner tags.
 
    Returns:
        processed_tags (List[List[str]]): list of processed ner tags, each element is a list of ner tags of a document
        ner_tokens (dic): key is ner tag, value is a list containing all tokens labeled as the tag  
        text (str): original text
    """
    # create new ner tags
    processed_tags = []
    text = []
    ner_tokens = {}
    for n in ner_tags:
        ner_tokens[n] = []
    if not sample:
        sample = len(data)
    for n in range(sample):
        doc = " ".join(data[n]['tokens'])
        ner_copy = data[n]['ner_tags'].copy()
        for i, t in enumerate(ner_copy):
            # taget ner
            if t in ner_tags:
                ner_tokens[t].append(data[n]['tokens'][i])
            # redundant ner
            elif t != "O":
                ner_copy[i] = "O"

        processed_tags.append(ner_copy)
        text.append(doc)

    return (processed_tags, ner_tokens, text)

def find_frequent_subword(tokens, n_gram, top):
    subwords = []
    for t in tokens:
        if len(t)>=n_gram:
            subwords.extend([t[i:i+n_gram] for i in range(len(t)-n_gram+1)])
    counts = Counter(subwords)
    top_subwords = counts.most_common(top)
    return top_subwords