
import torch
import torch.nn as nn

class PrmModel(nn.Module):
    def __init__(self, vocab_size,
                 embedding_size, hidden_size,
                 attn_hidden_size,
                 cls_hidden_size,
                 r_size,
                 num_labels,
                 num_layers=1,
                 dropout=0,
                 ):
        """
        Args:
            vocab_size: Number of words in the vocabulary.
            hidden_size: Number of dimensions of RNN hidden cell. # 512
            num_layers: The number of layers of the RNNs.

        """
        super(PrmModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_labels=num_labels
        # lstm layer

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout= dropout, batch_first=True,bidirectional=True)
        self.attention = Attention(hidden_size, attn_hidden_size, r_size)
        self.classifier = Classifier(cls_hidden_size, r_size, self.num_labels)
        self.max_pooling = nn.MaxPool1d(hidden_size * 2)


    def forward(self, x):
        embedding_states = self.embedding(x)  # embedding_states : (batch, length, 512)
        hidden_states, _ = self.lstm(embedding_states) # hidden_states : (batch, length, lstm_hidden_states * 2)

        attn_hidden, attn_weights = self.attention(hidden_states) # embs : (batch, r_size, lstm_hidden_states * 2)
        pooled = torch.squeeze(self.max_pooling(attn_hidden), 2) # pooled : (batch, r_size)
        outputs = self.classifier(pooled) # outputs_1 : (batch, n_class)

        return outputs,attn_weights



class Attention(nn.Module):
    def __init__(self, lstm_hidden_size, attn_hidden_size, r_size):
        super(Attention, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.attn_hidden_size = attn_hidden_size

        self.w_s1 = nn.Linear(self.lstm_hidden_size * 2, self.attn_hidden_size)
        self.w_s2 = nn.Linear(self.attn_hidden_size, r_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        hidden_states = x # hidden_states : (batch, length , lstm_hidden_size * 2)
        x = self.tanh(self.w_s1(x)) # x : (batch, length, attn_hidden_size)
        x = self.w_s2(x) # x : (batch, length, r_size)
        x = torch.permute(x, (0, 2, 1)) # x : (batch, r_size, length)
        annt_matrix = self.softmax(x)
        emb_matrix = torch.matmul(annt_matrix, hidden_states)

        return emb_matrix, annt_matrix

class Classifier(nn.Module):
    def __init__(self, hidden_size, prior_size, n_class):
        super(Classifier, self).__init__()

        self.ff = nn.Linear(prior_size, hidden_size)
        self.pred = nn.Linear(hidden_size, n_class)

        self.relu =nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.ff(x)) # x : (batch, ff_hidden_size)
        outputs = self.pred(x) # outputs : (batch, n_class)

        return self.softmax(outputs)
