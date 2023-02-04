# Performs 'Clean' operation
import pandas as pd

from config.props import stops
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import emoji

warnings.simplefilter(action='ignore', category=FutureWarning)
lemmatizer = WordNetLemmatizer()
tag_map = defaultdict(lambda: wordnet.NOUN)
tag_map['J'] = wordnet.ADJ
tag_map['V'] = wordnet.VERB
tag_map['R'] = wordnet.ADV


# idea: remove URL and USER totally
def preprocess(data, col_name, new_name):
    # preprocessing step
    data[new_name] = data[col_name].apply(emoji.demojize)
    data[new_name] = data[new_name].str.replace(r'(\$[A-Z]{2,})([a-z]+)', r'\1 \2', regex=True)
    data[new_name] = data[new_name].str.replace(r"\$[A-Z]+\b", ' ', regex=True)  # remove stock symbols
    data[new_name] = data[new_name].str.lower()
    data[new_name] = data[new_name].str.replace(r"(?!^no)(?!^up)(?![A-Z])^[\d\w]{1,2} ", ' ', regex=True)
    data[new_name] = data[new_name].str.replace(r"(?! no$)(?!up$)(?![A-Z]) [\d\w]{1,2}$", ' ', regex=True)
    data[new_name] = data[new_name].str.replace(r"(?! no )(?! up )(?![A-Z]) [\d\w]{1,2} ", ' ', regex=True)

    # handling links
    data[new_name] = data[new_name].str \
        .replace(
        r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)",
        " URL ", regex=True)

    data[new_name] = data[new_name].str.replace("&lt;", ' less than ', regex=True)
    data[new_name] = data[new_name].str.replace("&gt;", 'more than', regex=True)
    data[new_name] = data[new_name].str.replace("&le;", 'less or equal to', regex=True)
    data[new_name] = data[new_name].str.replace("&ge;", 'more or equal to', regex=True)

    data[new_name] = data[new_name].str.replace("`", '', regex=True)  # handle brackets, commas and other
    data[new_name] = data[new_name].str.replace("(", '', regex=False)
    data[new_name] = data[new_name].str.replace(")", '', regex=False)
    data[new_name] = data[new_name].str.replace("'", '', regex=False)
    data[new_name] = data[new_name].str.replace(":", ' ', regex=False)
    data[new_name] = data[new_name].str.replace(",", ' ', regex=True)
    data[new_name] = data[new_name].str.replace("*", ' ', regex=True)
    data[new_name] = data[new_name].str.replace("prev ", ' previous ', regex=False)
    data[new_name] = data[new_name].str.replace("yr.", ' year ', regex=False)
    data[new_name] = data[new_name].str.replace(" yr ", ' year ', regex=False)
    data[new_name] = data[new_name].str.replace(" yr.$", ' year ', regex=True)

    data[new_name] = data[new_name].replace(r' -([0-9])', r' negative \1',
                                            regex=True)  # handle negative numbers
    data[new_name] = data[new_name].replace(r'^-([0-9])', r'negative \1', regex=True)

    data[new_name] = data[new_name].replace(r' \+([0-9])', r' \1', regex=True)  # handle positive numbers
    data[new_name] = data[new_name].replace(r'^\+([0-9])', r'\1', regex=True)

    data[new_name] = data[new_name].replace(r'([0-9])-([0-9])', r'\1 to \2', regex=True)
    data[new_name] = data[new_name].replace(r'([0-9]) - ([0-9])', r'\1 to \2',
                                            regex=True)  # same but different comma
    data[new_name] = data[new_name].replace(r'(\w)-(\w)', r'\1 \2', regex=True)
    data[new_name] = data[new_name].replace(r'([0-9]),([0-9])', r'\1\2',
                                            regex=True)  # commas in numbers not needed
    data[new_name] = data[new_name].replace(r'([0-9]),([0-9])', r'\1\2',
                                            regex=True)  # commas in numbers not needed
    data[new_name] = data[new_name].replace(r'(.)(,)(.)', r'\1 \3',
                                            regex=True)  # commas excluding in numbers is blank space
    data[new_name] = data[new_name].replace(',', '', regex=True)

    data[new_name] = data[new_name].str.replace("@(\w){1,15}", ' USER  ', regex=True)

    data[new_name] = data[new_name].str.replace("%", ' percent ', regex=False)
    data[new_name] = data[new_name].replace(r"([0-9]+)m", r' \1 million ', regex=True)
    data[new_name] = data[new_name].replace(r"([0-9]+)b", r' \1 billion ', regex=True)

    data[new_name] = data[new_name].str.replace(r"[^a-zA-Z\d\s:\.]", ' ',
                                                regex=True)  # remove all non alphanumerical
    data[new_name] = data[new_name].str.replace(r"[\n]", ' ', regex=True)  # remove new lines

    data[new_name] = data[new_name].replace(r"coronavirus", 'coronavirus ', regex=True)
    data[new_name] = data[new_name].replace(r"RT s+ ", ' ', regex=True)

    data[new_name] = data[new_name].replace(r'([a-zA-z])([0-9])', r'\1 \2',
                                            regex=True)  # spaces between numbers and letters
    data[new_name] = data[new_name].replace(r'([0-9])([a-zA-Z])', r'\1 \2',
                                            regex=True)  # spaces between numbers and letters
    data[new_name] = data[new_name].replace(r'([0-9])([a-zA-Z])', r'\1 \2',
                                            regex=True)  # spaces between numbers and letters
    data[new_name] = data[new_name].replace(r'([0-9])([a-zA-Z])', r'\1 \2',
                                            regex=True)  # spaces between numbers and letters

    data[new_name] = data[new_name].replace(r"(\w)\1{3,}", r'\1', regex=True)  # get rid of repeating symbols
    data[new_name] = data[new_name].replace(r"(\.){2,}", ' ', regex=True)
    data[new_name] = data[new_name].replace(r"(\w+)(\.) ", r'\1 ', regex=True)  # get rid of dots in the middle
    data[new_name] = data[new_name].replace(r'( \.)+', r' ', regex=True)

    # remove stop words
    for word in stops:
        data[new_name] = data[new_name].replace(' ' + word + ' ', ' ', regex=True)
        data[new_name] = data[new_name].replace(word + r'$', ' ', regex=True)

    data[new_name] = data[new_name].replace(r"\s+|\s+$|\s+(?=\s)", ' ', regex=True)  # whitespaces
    data[new_name] = data[new_name].str.strip(' ,?!.():;-')  # strip
    data = data[(data['sentiment'] == 1) | (data['sentiment'] == 0) | (data['sentiment'] == -1)]
    data.dropna(subset=[new_name], inplace=True)

    return data


def get_wordnet_pos(word):  # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.ADJ)


def lemmatize_entry(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens)]
    return " ".join(lemmatized_tokens)


# for live processing
def get_preprocessed_entry(entry):
    d = {'text': [entry], 'sentiment': [1]}
    data = pd.DataFrame(data=d)
    data['sentiment'] = data['sentiment'].astype('int')

    preprocess(data, 'text', 'clean_text')
    data['clean_text'] = data['clean_text'].apply(lemmatize_entry)
    return data
