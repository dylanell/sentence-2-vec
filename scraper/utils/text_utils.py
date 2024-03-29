"""
General text processing utilities.
"""

import nltk
from nltk.corpus import stopwords
import unidecode
import re
import inflect

# initialize inflect library engine
inflect_engine = inflect.engine()

# NOTE: uncomment below for punctuation filtering
# import string
# make punctuation translation table
# punc_table = str.maketrans('', '', string.punctuation)
# we actually want to translate a space for '-' instead of None
# punc_table[ord('-')] = ' '

# flattens a list containing other nested lists
def flatten_list(list_object):
    rt = []
    for i in list_object:
        if isinstance(i, list):
            rt.extend(flatten_list(i))
        else:
            rt.append(i)
    return rt


# checks if string 's' is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# convert contractions back to their full words
def decontract_text(text):
    # specific
    text = re.sub(r'won\'t', 'will not', text)
    text = re.sub(r'can\'t', 'can not', text)
    text = re.sub(r'y\'all', 'you all', text)

    # general
    text = re.sub(r'n\'t', ' not', text)
    text = re.sub(r'\'re', ' are', text)
    text = re.sub(r'\'s', ' is', text)
    text = re.sub(r'\'d', ' would', text)
    text = re.sub(r'\'ll', ' will', text)
    text = re.sub(r'\'t', ' not', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'m', ' am', text)
    return text


# process alphanumeric (number string) text
def process_alnumeric(s):
    # convert number string to words
    text = inflect_engine.number_to_words(s)

    # lowercase all words
    text = text.lower()

    # strip '-'
    text = text.replace('-', ' ')

    # split text into work tokens
    tokens = nltk.tokenize.word_tokenize(text)

    return tokens


# performs normalization and preprocessing on the text
def process_text(text):
    # convert special characters to their base form
    text = unidecode.unidecode(text)

    # lowercase all words
    text = text.lower()

    # decontract any contractions
    text = decontract_text(text)

    # split text into work tokens
    tokens = nltk.tokenize.word_tokenize(text)

    # find words separated by ('.', '!', '?') without a space
    tokens = flatten_list(list(map(
        lambda x: [x.split('.')[0], '.', x.split('.')[-1]]
        if ('.' in x and not is_number(x) and len(x) > 1) else x, tokens)))
    tokens = flatten_list(list(map(
        lambda x: [x.split('!')[0], '.', x.split('.')[-1]]
        if ('.' in x and not is_number(x) and len(x) > 1) else x, tokens)))
    tokens = flatten_list(list(map(
        lambda x: [x.split('?')[0], '.', x.split('.')[-1]]
        if ('.' in x and not is_number(x) and len(x) > 1) else x, tokens)))

    # convert numbers to nested word tokens
    tokens = flatten_list(list(map(
        lambda x: process_alnumeric(x) if is_number(x) else x, tokens)))

    # remove non alpha-numeric
    tokens = [x for x in tokens if x.isalpha()]

    # remove stopwords
    tokens = [x for x in tokens if x not in stopwords.words('english')]

    # stem word tokens
    stemmer = nltk.stem.PorterStemmer()
    tokens = list(map(lambda x: stemmer.stem(x), tokens))

    # re-join tokens
    joined_tokens = ' '.join(tokens)

    return joined_tokens

def test_qa_is_good(q, a):
    # reject if question or answer is None
    if (q is None) or (a is None):
        return False

    # reject if answer contains a link
    link_flags = ['http', 'https', 'www', '.net', '.com', '.org', '.gov']
    if any(flag in a.text for flag in link_flags):
        return False

    # if we got here qa pair is all good
    return True

