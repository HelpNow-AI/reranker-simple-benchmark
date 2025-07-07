from .base import BaseReranker
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3ForCausalLM

class Qwen3Reranker(BaseReranker):
    def __init__(self, model_path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.reranker = Qwen3ForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, attn_implementation="flash_attention_3").to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output
    
    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.reranker.device)
        return inputs

    def compute_score_batch(self, query: str, docs: list[str], normalize: bool = False) -> list[float]:
        pairs = [self.format_instruction(self.task, query, doc) for doc in docs]
        inputs = self.process_inputs(pairs)
        batch_scores = self.reranker(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def compute_score(self, pairs: list[tuple[str, str]], normalize: bool = True) -> list[float]:
        pass