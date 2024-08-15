from nltk_summarization import NLTKTextSummarizer
from sumy_summarization import SUMYTextSummarizer
from bert_extractive_summarization import BertExtSummarizer
from bart_abstractive_summarization import BartAbsSummarizer
import yaml
import spacy

### Reading the config yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

### Read the input text file and save the text content in the config dictioanry.
try:
    with open(config['text_file'], 'r', encoding='utf-8') as file:
        config['text'] = file.read()
except FileNotFoundError:
    print(f"Error: File '{config['text_file']}' not found.")
except IOError as e:
    print(f"Error reading file: {e}")


algorithm_dict = {'NLTK':NLTKTextSummarizer,
              'SUMY':SUMYTextSummarizer,
              'BERTEXT':BertExtSummarizer,
              'BARTABS':BartAbsSummarizer,
                }

if __name__ == "__main__":
    ### Calculate the number of sentences in the summary and the percentage of the original text it represents.
    if config['SUMMARY_LENGTH']['type'] == "PERCENTAGE":
        config["SENTENCES_COUNT"] = int(len(list(spacy.load("en_core_web_sm")(config['text']).sents)) * (config['SUMMARY_LENGTH']['value']) / 100)
        config["PERCENT"] = (config['SUMMARY_LENGTH']['value']) / 100
    elif config['SUMMARY_LENGTH']['type'] == "COUNT":
        config["SENTENCES_COUNT"] = config['SUMMARY_LENGTH']['value']
        config["PERCENT"] = (config['SUMMARY_LENGTH']['value']) / len(list(spacy.load("en_core_web_sm")(config['text']).sents))
   
    ### Loop through each algorithm in config file prepare the summary and save the results to the output file.
    for algo in config['algorithms']:
        summarizer = algorithm_dict[algo](config)
        summaries = summarizer.generate_summaries()

        with open('summaries.txt', 'a', encoding='utf-8') as file:
                for key, summary in summaries.items():
                    file.write(f"**{key}:**\n{summary}\n\n")
        
        print(f"Summaries for {algo} saved to 'summaries.txt'")
