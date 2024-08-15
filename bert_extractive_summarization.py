from summarizer import Summarizer

class BertExtSummarizer:
    def __init__(self,config):
        self.config = config        
        self.text = self.config.get('text', '')
        self.min_length = self.config.get('min_length')
        self.max_length = self.config.get('max_length')
        self.sentences_count = self.config.get('SENTENCES_COUNT')
    
    def generate_summaries(self):
        ### Generate a summary using BERT with length constraints or sentence count based on the parameters provided in config file
        summarizer = Summarizer()
        if all([self.min_length, self.max_length]):
            summary = summarizer(self.text, min_length=self.min_length, max_length=self.max_length) 
        else:
            summary = summarizer(self.text, num_sentences=self.sentences_count) 
        return {'BERT Extractive Summarization':summary}