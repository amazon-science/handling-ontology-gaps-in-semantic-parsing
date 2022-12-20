'''
This code starts from the code of KQA-Pro_Baseline" (https://github.com/shijx12/KQAPro_Baselines) at the commit "7cea2738fd095a2c17594d492923ee80a212ac0f (4th October 2022) then some modifications are applied.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''
import json
import pickle
import torch
from utils.misc import invert_dict

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

def collate(batch):

    
    batch = list(zip(*batch))


    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    #if batch[-1][0] is None: andbac modification: trick to understand if is in test mode
    if batch[-2][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    impossible_to_answer = torch.stack(batch[5])  # andbac: batch = 256 so the shape will be torch.Size([256, 1])
    return source_ids, source_mask, choices, target_ids, answer, impossible_to_answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers, self.impossible_to_answer = inputs
        print("impossible_to_answer", self.impossible_to_answer)
        #self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers = inputs # andbac
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        impossible_to_answer = torch.BoolTensor([self.impossible_to_answer[index]])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])

        return source_ids, source_mask, choices, target_ids, answer, impossible_to_answer


    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(6): # andbac was 5 here
                x = pickle.load(f)
                inputs.append(x)

        dataset = Dataset(inputs)


        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab