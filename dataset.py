from torch.utils.data import Dataset
import torch


class Shakespeare(Dataset):
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            text = f.read()

      
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        self.data = [self.char_to_idx[char] for char in text]
        self.seq_length = 30

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]

        target = self.data[idx + 1:idx + self.seq_length + 1]

        input_tensor = torch.tensor(input_seq, dtype=torch.long)

        return input_tensor, torch.tensor(target, dtype=torch.long)

if __name__ == '__main__':
    input_file = "shakespeare_train.txt"  # Path to the Shakespeare dataset
    shakespeare_dataset = Shakespeare(input_file)

    print("Total characters:", len(shakespeare_dataset.data))
    print("Total unique characters:", len(shakespeare_dataset.char_to_idx))

    input_seq, target = shakespeare_dataset[0]

    input_seq = [idx.item() for idx in input_seq]
    target = [idx.item() for idx in target]

    input_text = ''.join([shakespeare_dataset.idx_to_char[idx] for idx in input_seq])
    target_text = ''.join([shakespeare_dataset.idx_to_char[idx] for idx in target])
   
