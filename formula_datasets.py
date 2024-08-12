import torch
from torch.nn.utils.rnn import pad_sequence

class LoadedDataset:
    """assumes each line in dataset file is an example of the form '<func> ===> <ans>' \n
    returns lists [<func>, <ans>] from the first train_size_max examples in file"""
    def __init__(self, train_fname:str, train_size_max:int, offset=0):
        with open(train_fname, "r") as f:
            self.lines = []
            offset_remaining = offset
            while offset_remaining > 0:
                newlines = f.readlines(offset_remaining)
                if len(newlines) == 0:
                    return
                offset_remaining -= len(newlines)
            loaded = 0
            while loaded < train_size_max:
                l = f.readline()
                if not l:
                    break
                l = l.strip()
                if l != "" and "===>" in l:
                    self.lines.append(l.split("===>"))
                    loaded += 1
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, i):
        return self.lines[i]

def query_instruction(sample):
    return f"[INST] Integrate {sample} [/INST]"

class TokenizerDataset:
    """returns a tensor of tokens from example '[INST] Integrate <func> [/INST] <ans>' with bos_id at the start and eos_id at the end. \n
    <func> and <ans> are strings obtained from input_dataset in [<func>, <ans>] list format. \n
    also returns length of the prompt before the first token of <ans>"""
    def __init__(self, input_dataset, input_tokenizer):
        self.dataset = input_dataset
        self.tok = input_tokenizer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        func, integ = self.dataset[i]
        tokens = [self.tok.bos_token_id] + self.tok.encode(query_instruction(func), add_special_tokens=False)
        L = len(tokens)
        tokens += self.tok.encode(integ, add_special_tokens=False) + [self.tok.eos_token_id]
        return torch.LongTensor(tokens), L

class PromptTokenizerDataset:
    def __init__(self, input_dataset, input_tokenizer):
        self.dataset = input_dataset
        self.tok = input_tokenizer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        prompt, ans = self.dataset[i]
        tokens = [self.tok.bos_token_id] + self.tok.encode(query_instruction(prompt), add_special_tokens=False)
        L = len(tokens)
        return torch.LongTensor(tokens), L, prompt, ans

def train_collate(x):
    """returns batch of left-padded examples, \n
    the labels where padding tokens (from the left) are masked, \n
    and a list of example lengths"""
    ignore_id = -100
    padded = pad_sequence([tup[0].flip(dims=[0]) for tup in x], batch_first=True, padding_value=2).flip(dims=[1])
    lens = [tup[1] for tup in x]
    labels = padded.clone()
    maxlen = padded.shape[1]
    for i_len, L in enumerate(lens):
        ex_len = x[i_len][0].shape[0]
        labels[i_len, :maxlen-ex_len] = torch.full((maxlen-ex_len,), ignore_id)

    return padded, labels, lens

def prompt_eval_collate(x):
    """returns left-padded tokenized prompts, their lengths, prompts and answers in the form of Tensor, list[int], list(str), list(str).\n
    expects iterable of tuples (prompt_tensor, prompt_len, prompt_str, ans_str)"""
    padded = pad_sequence([tup[0].flip(dims=[0]) for tup in x], batch_first=True, padding_value=2).flip(dims=[1])
    lens = [tup[1] for tup in x]
    funcs = [tup[2] for tup in x]
    ans = [tup[3] for tup in x]

    return padded, lens, funcs, ans