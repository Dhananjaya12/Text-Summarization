from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class BartAbsSummarizer:
    def __init__(self,config):
        self.config = config        
        self.text = self.config.get('text', '')
        self.bart_abs_model = self.config.get('bart_abs_model')
        self.percent = self.config.get('PERCENT')
        self.min_length = self.config.get('min_length')
        self.max_length = self.config.get('max_length')

    def summarize_text(self, model_name):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        # Tokenize without truncation
        inputs_no_trunc = tokenizer(self.text, max_length=None, return_tensors='pt', truncation=False)

        # Get batches of tokens corresponding to the exact model_max_length
        chunk_start = 0
        chunk_end = 1024 #tokenizer.model_max_length  # == 1024 for Bart
        inputs_batch_lst = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # Get batch of n tokens
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            chunk_start += 1024 #tokenizer.model_max_length  # == 1024 for Bart
            chunk_end += 1024 #tokenizer.model_max_length  # == 1024 for Bart
        
        # Define max_output_length and min_output_length dynamically
        if all([self.min_length, self.max_length]):
            max_output_length = self.max_length
            min_output_length = self.min_length
        else:
            max_output_length = int(1024 * self.percent)
            min_output_length = int(1024 * (self.percent - 0.01))
            
        # Generate a summary on each batch
        summary_ids_lst = [model.generate(inputs, num_beams=4,min_length = min_output_length, max_length=max_output_length, early_stopping=True) for inputs in inputs_batch_lst]

        # Decode the output and filter relevant summaries
        summary_batch_lst = []
        for summary_id in summary_ids_lst:
            summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary_batch_lst.append(summary_batch[0])

        # Join relevant summaries into one string with one paragraph per summary batch
        summary_all = '\n'.join(summary_batch_lst)
        return summary_all
    
    def generate_summaries(self):
        summaries = {}
        for model in self.bart_abs_model:
            summaries[model] = self.summarize_text(model)
        return summaries
        
    