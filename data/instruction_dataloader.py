# load the json 
# template: json dict -> text + give a prompt
# dataset: one instruction + 1 response = one token list 
# collate: add <end>, pad, shift by one, mask with -100

import json
import torch 
import tiktoken 
from torch.utils.data import Dataset, DataLoader 

#format_input: convert the dict into a prompt with #instruction and #input
def format_input(entry):
    text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    if entry["input"]:
        text += f"\n\n### Input:\n{entry['input']}"
    return text 

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encoded = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded.append(tokenizer.encode(full_text)) # (1 input + 1 output) = 1 pair 
    def __len__(self):
        return len(self.encoded)
    def __getitem__(self, index):
        return self.encoded[index]

def collate_fn(batch, pad_id = 50256, ignore_index=-100):
    #find the target length (max) 
    #loop over each example to add padding id  [1,2,3,E,E]
    #shift by one, (inp and tgt)
    #mask the extra padding (convert extra padding to -100) [1,2,3,E,-100]
    
    #find the target length (max) 
    batch_max = max(len(item) for item in batch)+1 #+1 for the End token 

    #loop over each example and add pad id 
    input = []
    target = []
    for item in batch:
        x = item + [pad_id]
        while len(x) < batch_max:
            x = x + [pad_id]

        #inp and tgt
        inp = torch.tensor(x[:-1])
        tgt = torch.tensor(x[1:])

        #mask the extra padding 
        pad_pos = torch.nonzero(tgt==pad_id).squeeze(-1)
        if len(pad_pos) > 1:
            tgt[pad_pos[1:]] = ignore_index
        
        input.append(inp) 
        target.append(tgt) 
    return torch.stack(input), torch.stack(target) 

    #input 
    # [tensor([1, 2, 3, 4, 5]),
    #  tensor([6, 7, 8, E, E])]

    #torch.stack(input)
    # tensor([[    1,     2,     3,     4,     5],
    #         [    6,     7,     8, 50256, 50256]])
    




    
