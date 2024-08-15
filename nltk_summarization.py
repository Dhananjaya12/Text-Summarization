from string import punctuation
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

class NLTKTextSummarizer:
    def __init__(self, config):
        self.config = config
        self.text = self.config.get('text', '')
        self.sentences_count = self.config.get('SENTENCES_COUNT')
        self.sentences, self.freq_dist = self.preprocessing()
        self.normalized_freq_dist = self.normalized_word_freq(self.freq_dist)
        self.non_normalized_sent_scores = self.scoring_sentences(self.freq_dist)
        self.normalized_sent_scores = self.scoring_sentences(self.normalized_freq_dist)
    
    def preprocessing(self):
        ### Tokenize the text into sentences and words
        sentences = sent_tokenize(self.text)
        words = word_tokenize(self.text.lower())
        ### Filter out stopwords and punctuations
        stopwords_list = set(stopwords.words('english') + list(punctuation))
        filtered_words = [word for word in words if word not in stopwords_list]
        ### Calculate word frequencies
        freq_dist = FreqDist(filtered_words)
        return sentences, freq_dist

    def normalized_word_freq(self, freq_dist):
        ### Normalize word frequencies by dividing by the maximum frequency
        maximum_frequency = max(freq_dist.values())
        normalized_freq_dist = {word: freq / maximum_frequency for word, freq in freq_dist.items()}
        return normalized_freq_dist

    def scoring_sentences(self, freq_dist):
        ### Score sentences based on the sum of word frequencies
        sent_scores = {}
        for i, sentence in enumerate(self.sentences):
            for word in word_tokenize(sentence.lower()):
                if word in freq_dist:
                    if i in sent_scores:
                        sent_scores[i] += freq_dist[word]
                    else:
                        sent_scores[i] = freq_dist[word]
        return sent_scores

    def preparing_summary_number(self, sent_scores):
        ### Select top-ranked sentences for the summary and combine them into a text
        sorted_sentences = sorted(sent_scores, key=sent_scores.get, reverse=True)
        summary_sentences = sorted(sorted_sentences[:self.sentences_count])
        summary_text = ' '.join([self.sentences[i] for i in summary_sentences])
        return summary_text

    def preparing_summary_avg(self, sent_scores):
        ### Prepare summary using sentences with scores above the average of sentence scores
        sum_values = sum(sent_scores.values())
        avg = sum_values / len(sent_scores)
        summary_text = ' '.join([self.sentences[i] for i in sent_scores if sent_scores[i] > 1.2 * avg])
        return summary_text

    def generate_summaries(self):
        # Summaries using non-normalized frequencies
        summary_by_number_non_normalized = self.preparing_summary_number(self.non_normalized_sent_scores)
        summary_by_avg_non_normalized = self.preparing_summary_avg(self.non_normalized_sent_scores)

        # Summaries using normalized frequencies
        summary_by_number_normalized = self.preparing_summary_number(self.normalized_sent_scores)
        summary_by_avg_normalized = self.preparing_summary_avg(self.normalized_sent_scores)

        return {
            'Summary by Number of Sentences (Non-Normalized)': summary_by_number_non_normalized,
            'Summary by Average Scoring (Non-Normalized)': summary_by_avg_non_normalized,
            'Summary by Number of Sentences (Normalized)': summary_by_number_normalized,
            'Summary by Average Scoring (Normalized)': summary_by_avg_normalized
        }
