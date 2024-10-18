import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from scipy.special import expit
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.attention_weights_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden_state, encoder_outputs):
        batch_size, seq_len, N = encoder_outputs.size()
        decoder_hidden_state = decoder_hidden_state.view(batch_size, 1, N).repeat(1, seq_len, 1)
        combined_input = torch.cat((encoder_outputs, decoder_hidden_state), dim=2)
        combined_input = combined_input.view(batch_size * seq_len, -1)

        attention_values = self.fc1(combined_input)
        attention_values = self.fc2(attention_values)
        attention_values = self.fc3(attention_values)
        attention_values = self.fc4(attention_values)
        
        attention_weights = self.attention_weights_layer(attention_values).view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector
                
class Encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512):
        super(Encoder, self).__init__()
        self.compress = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, feature_size = x.size()
        x = self.compress(x.view(-1, feature_size)).view(batch_size, seq_len, -1)
        x = self.dropout(x)
        return self.gru(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, word_dim, dropout_percentage=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.gru = nn.GRU(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_hidden, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_hidden.size()
        decoder_hidden = None if encoder_hidden is None else encoder_hidden
        decoder_input = Variable(torch.ones(batch_size, 1)).long().cuda()
        seq_log_prob = []

        if mode == 'train':
            targets = self.embedding(targets)
            for i in range(targets.size(1) - 1):  
                threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
                if random.uniform(0.05, 0.995) > threshold:
                    current_input_word = targets[:, i]  
                else: 
                    current_input_word = self.embedding(decoder_input).squeeze(1)
                context = self.attention(decoder_hidden, encoder_output)
                gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
                gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
                log_prob = self.to_final_output(gru_output.squeeze(1))
                seq_log_prob.append(log_prob.unsqueeze(1))
                decoder_input = log_prob.unsqueeze(1).max(2)[1]
            seq_log_prob = torch.cat(seq_log_prob, dim=1)
        return seq_log_prob, seq_log_prob.max(2)[1]

    def infer(self, encoder_hidden, encoder_output, max_len=28):
        _, batch_size, _ = encoder_hidden.size()
        decoder_hidden = None if encoder_hidden is None else encoder_hidden
        decoder_input = Variable(torch.ones(batch_size, 1)).long().cuda()
        seq_log_prob = []

        for _ in range(max_len):
            current_input_word = self.embedding(decoder_input).squeeze(1)
            context = self.attention(decoder_hidden, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
            log_prob = self.to_final_output(gru_output.squeeze(1))
            seq_log_prob.append(log_prob.unsqueeze(1))
            decoder_input = log_prob.unsqueeze(1).max(2)[1]
        seq_log_prob = torch.cat(seq_log_prob, dim=1)

        return seq_log_prob, seq_log_prob.max(2)[1]


    def beam_search(self, encoder_hidden, encoder_outputs, beam_width=3, max_seq_len=28):
        batch_size = encoder_hidden.size(1)
        beam_hidden = encoder_hidden.repeat(1, beam_width, 1)
        beam_input = torch.ones(batch_size, 1).long().cuda()
    
        beam_scores = torch.zeros(batch_size, beam_width).cuda()
        beam_sequences = torch.ones(batch_size, beam_width, max_seq_len).long().cuda()
    
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, beam_width, 1, 1).view(batch_size * beam_width, -1, encoder_outputs.size(-1))
    
        for step in range(1, max_seq_len):
            if step == 1:
                context = self.attention(beam_hidden, encoder_outputs)
            else:
                context = self.attention(beam_hidden.view(batch_size, beam_width, -1), encoder_outputs)
    
            gru_input = torch.cat((self.embedding(beam_input).squeeze(1), context), dim=1).unsqueeze(1)
            output, beam_hidden = self.gru(gru_input, beam_hidden)
            logits = self.to_final_output(output.squeeze(1))
    
            scores, indices = torch.topk(F.log_softmax(logits, dim=1), beam_width, dim=1)
            beam_scores = beam_scores.unsqueeze(2) + scores
    
            if step == 1:
                beam_scores = beam_scores.squeeze(1)
                beam_hidden = beam_hidden.repeat(1, beam_width, 1)
            else:
                beam_scores = beam_scores.view(batch_size, -1)
    
            topk_scores, topk_indices = torch.topk(beam_scores, beam_width, dim=1)
            prev_beams = topk_indices // beam_width
            next_tokens = topk_indices % beam_width
    
            beam_hidden = beam_hidden.view(batch_size, beam_width, -1).gather(1, prev_beams.unsqueeze(2).repeat(1, 1, beam_hidden.size(2)))
            beam_sequences = beam_sequences.gather(1, prev_beams.unsqueeze(2).repeat(1, 1, beam_sequences.size(2)))
            beam_sequences[:, :, step] = indices.view(batch_size, beam_width)[torch.arange(batch_size).unsqueeze(1), prev_beams]
    
            beam_input = indices.view(batch_size, beam_width)[torch.arange(batch_size), next_tokens].unsqueeze(1)
            beam_hidden = beam_hidden.view(batch_size * beam_width, -1)
    
        best_sequences = beam_sequences[torch.arange(batch_size), torch.argmax(topk_scores, dim=1)]
        return best_sequences


    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode='train', target_sentences=None, tr_steps=None):
        encoder_output, encoder_hidden = self.encoder(avi_feat)
        if mode == 'train':
            return self.decoder(encoder_hidden, encoder_output, targets=target_sentences, mode=mode, tr_steps=tr_steps)
        elif mode == 'inference':
            return self.decoder.infer(encoder_hidden, encoder_output)
        return self.decoder.beam_search(encoder_hidden, encoder_output)
