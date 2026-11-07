from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []

        #tokenize the entire text 
        token_id = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        #use a sliding window to chunk the text into input and target tensors
        for i in range(0,len(token_id)-max_length,stride):
            input = token_id[i:i+max_length] #i+max_length is exclusive 
            target = token_id[i+1:i+max_length+1] #i+max_length+1 is exclusive
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, key):
        return self.input_ids[key], self.target_ids[key]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    #initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #create dataset 
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    #create dataloader 
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers 
    )
    return dataloader
            