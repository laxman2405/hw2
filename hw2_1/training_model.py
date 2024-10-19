import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
import random
import os
from torch.utils.data import Dataset, DataLoader
import seq_2_seq_model 
from torch.autograd import Variable
from bleu_eval import BLEU
import sys
import time
from process_videos import get_mappings
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainingData(Dataset):
    def __init__(self, label_file, files_dir, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_to_index = w2i
        self.data_pair, self.avi = self._process_data(label_file)

    def _process_data(self, label_json):
        annotated_captions = []
        with open(label_json, 'r') as f:
            labels = json.load(f)

        for data in labels:
            for caption in data['caption']:
                tokenized_caption = [self.word_to_index.get(word, 3) for word in re.sub(r'[.!,;?]', ' ', caption).split()]
                tokenized_caption = [1] + tokenized_caption + [2]
                annotated_captions.append((data['id'], tokenized_caption))

        avi_features = {file.split('.npy')[0]: np.load(os.path.join(self.files_dir, file)) for file in os.listdir(self.files_dir)}
        return annotated_captions, avi_features

    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)

def training_batch_data(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def train_model(model, train_loader, loss_fn):
    epoch = 1
    loss_history = []

    while epoch <= 200:
        model.train()
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        total_loss = 0
        for idx, (avi_feats, ground_truths, lengths) in enumerate(train_loader):
            avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            optimizer.zero_grad()
            seq_logProb, _ = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
    
            ground_truths = ground_truths[:, 1:]
            loss = 0
    
            for i in range(len(seq_logProb)):
                seq_len = lengths[i] - 1 
                loss += loss_fn(seq_logProb[i, :seq_len], ground_truths[i, :seq_len])
    
            loss /= len(seq_logProb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch}/200: Average Loss: {avg_loss:.3f}")
        loss_history.append(avg_loss)
        epoch += 1
    return loss_history

def main():
    model_path = 'hw2_model_laxman.h5'
    training_feat_folder = sys.argv[1]
    training_label_json = sys.argv[2]
       
    word_to_index, index_to_word = get_mappings(training_feat_folder, training_label_json,"train")
        
    pickle.dump(word_to_index, open('word_to_index.obj', 'wb'))
    pickle.dump(index_to_word, open('index_to_word.obj', 'wb'))
        
    train_dataset = TrainingData(training_label_json, training_feat_folder, word_to_index)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=256, shuffle=True, num_workers=8, collate_fn=training_batch_data)
        
    encoder = seq_2_seq_model.Encoder()
    decoder = seq_2_seq_model.Decoder(512, len(index_to_word)+4, 1024, 0.3)
    model = seq_2_seq_model.Seq2Seq(encoder = encoder,decoder = decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_index['<pad>'])
    
    start = time.time()
    loss_history = train_model(model, train_dataloader, criterion)
    end = time.time()
    torch.save(model, model_path)
    with open('calculated_losses.txt', 'w') as f:
        for loss in loss_history:
            f.write(f"{loss}\n")
    print("Training completed in {} seconds.\n".format(end-start))
    
if __name__ == '__main__':
        main()