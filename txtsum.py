from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request

fetched_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Joe_Biden')
article_read = fetched_data.read()
article_parsed = BeautifulSoup.BeautifulSoup(article_read, 'html.parser')
paragraphs = article_parsed.find_all('p')
article_content = ''
for p in paragraphs:
    article_content += p.text

def _create_dictionary_table(text_string) -> dict:
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    stem = PorterStemmer()
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table

def _calculate_sentence_scores(sentences, frequency_table) -> dict:
    sentence_weight = dict()
    for sentence in sentences:
        sentence_wordcount = (len(word_tokenize(sentence)))
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sentence[:7] in sentence_weight:
                    sentence_weight[sentence[:7]] += frequency_table[word_weight]
                else:
                    sentence_weight[sentence[:7]] = frequency_table[word_weight]
        sentence_weight[sentence[:7]] = sentence_weight[sentence[:7]] / sentence_wordcount_without_stop_words

    return sentence_weight

def _calculate_average_score(sentence_weight) -> int:
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]
    average_score = (sum_values / len(sentence_weight))
    return average_score

def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''
    for sentence in sentences:
        if sentence[:7] in sentence_weight and sentence_weight[sentence[:7]] >= (threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary

def _run_article_summary(article):
    frequency_table = _create_dictionary_table(article)
    sentences = sent_tokenize(article)
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)
    threshold = _calculate_average_score(sentence_scores)
    article_summary = _get_article_summary(sentences, sentence_scores, 1.5 * threshold)

    return article_summary

if __name__ == '__main__':
    summary_results = _run_article_summary(article_content)
    print(summary_results)