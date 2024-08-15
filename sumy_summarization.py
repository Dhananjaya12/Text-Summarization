from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

class SUMYTextSummarizer:
    def __init__(self, config):
        self.config = config        
        self.text = self.config.get('text', '')
        self.sentences_count = self.config.get('SENTENCES_COUNT')
        self.language = self.config.get('LANGUAGE', 'english')
        self.bonus_words = self.config.get('bonus_words', [])
        self.stigma_words = self.config.get('stigma_words', [])
        self.null_words = self.config.get('null_words', [])

        self.algorithms = {
            'Edmundson': EdmundsonSummarizer,
            'LexRank': LexRankSummarizer,
            'Lsa': LsaSummarizer,
            'Luhn': LuhnSummarizer,
            'TextRank': TextRankSummarizer,
            'KL': KLSummarizer
        }

    def summarize_text(self, algorithm_name):
        ### Parse the input text and initialize the stemmer
        parser = PlaintextParser.from_string(self.text, Tokenizer(self.language))
        stemmer = Stemmer(self.language)

        algorithm_class = self.algorithms.get(algorithm_name)
        if not algorithm_class:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in available algorithms.")

        ### Initialize the summarizer with the chosen algorithm and configure it
        summarizer = algorithm_class(stemmer)
        summarizer.stop_words = get_stop_words(self.language)
        summarizer.bonus_words = self.bonus_words
        summarizer.stigma_words = self.stigma_words
        summarizer.null_words = self.null_words

        ### Generate the summary and by joining the sentences
        summary = summarizer(parser.document, self.sentences_count)
        return ' '.join(str(sentence) for sentence in summary)

    def generate_summaries(self):
        ### Generate and return summaries using all specified or default algorithms
        summaries = {}
        for algorithm_name in self.config.get('sumy_algorithms',self.algorithms.keys()):
            summaries[algorithm_name] = self.summarize_text(algorithm_name)
        return summaries