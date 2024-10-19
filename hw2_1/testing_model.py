import sys
import numpy as np
import torch
import json
import pickle as pk
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from bleu_eval import BLEU
import seq_2_seq_model
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class TestDataset(Dataset):
    def __init__(self, test_data_path):
        self.avi_features = [(file.split('.npy')[0], np.load(os.path.join(test_data_path, file))) for file in os.listdir(test_data_path)]
    
    def __len__(self):
        return len(self.avi_features)
    
    def __getitem__(self, idx):
        return self.avi_features[idx]

def compute_bleu_score(test_labels, predictions):
    bleu_scores = [BLEU(predictions[item['id']], [x.rstrip('.') for x in item['caption']], True) for item in test_labels]
    return sum(bleu_scores) / len(bleu_scores)

def test_model(test_loader, model, index_to_word, label_json, output_file, use_beam_search=False):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for idx, (video_ids, avi_feats) in enumerate(test_loader):
            avi_feats = avi_feats.cuda()
            avi_feats = Variable(avi_feats).float()

            _, predicted_sequences = model(avi_feats, mode='inference')                
            captions = [' '.join([index_to_word[token.item()] for token in seq if index_to_word[token.item()] not in ['<pad>', '<bos>', '<eos>','<unk>']])
                        for seq in predicted_sequences]
            
            for i, caption in enumerate(captions):
                predictions[video_ids[i]] = caption
    
    with open(output_file, 'w') as f:
        for video_id, caption in predictions.items():
            f.write(f"{video_id},{caption}\n")

    with open(label_json, 'r') as f:
        test_labels = json.load(f)
    average_bleu = compute_bleu_score(test_labels, predictions)
    print(f"Average BLEU score: {average_bleu}")

def main():
    test_feat_folder = sys.argv[1]
    output_file = sys.argv[2]
    test_label_json = 'testing_label.json'
    model_path = 'hw2_model_laxman.h5'

    # Load preprocessed vocab
    with open('index_to_word.obj', 'rb') as f:
        index_to_word = pk.load(f)
    
    vocab_size = len(index_to_word) + 4
    test_dataset = TestDataset(test_feat_folder)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

    # Load and evaluate the model
    model = torch.load(model_path)
    test_model(test_loader, model, index_to_word, test_label_json, output_file)

if __name__ == '__main__':
    main()
