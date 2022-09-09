
import torch
import torch.nn as nn

class AuxBasedPrmModel(nn.Module):
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
        super(AuxBasedPrmModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_labels=num_labels
        # lstm layer
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout= dropout, batch_first=True,bidirectional=True)
        self.attention = Attention(hidden_size, attn_hidden_size, r_size)
        self.classifier = PrmClassifier(cls_hidden_size, r_size, self.num_labels)
        self.max_pooling = nn.MaxPool1d(hidden_size * 2)


    def forward(self, x):
        embedding_states = self.embedding(x)  # embedding_states : (batch, length, 512)
        hidden_states, _ = self.lstm(embedding_states) # hidden_states : (batch, length, lstm_hidden_states * 2)

        attn_hidden, attn_weights = self.attention(hidden_states) # embs : (batch, r_size, lstm_hidden_states * 2)
        pooled = torch.squeeze(self.max_pooling(attn_hidden), 2) # pooled : (batch, r_size)
        outputs = self.classifier(pooled) # outputs_1 : (batch, n_class)

        return outputs,attn_weights

class AuxModel(nn.Module):
    def __init__(self, vocab_size,
                 embedding_size, hidden_size,
                 attn_hidden_size,
                 cls_hidden_size,
                 r_size,
                 num_categories,
                 num_layers=1,
                 dropout=0,
                 ):
        """
        Args:
            vocab_size: Number of words in the vocabulary.
            hidden_size: Number of dimensions of RNN hidden cell. # 512
            num_layers: The number of layers of the RNNs.

        """
        super(AuxModel, self).__init__()

        self.num_categories=num_categories
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # lstm layer

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout= dropout, batch_first=True,bidirectional=True)
        # for i in range(self.num_categories):
        #      globals()['self.attention_'+str(i)] = Attention(hidden_size, attn_hidden_size, r_size).to(device)
        #      globals()['self.classifier_'+str(i)] = Classifier(cls_hidden_size, r_size, 2).to(device)
        
        #self.attention_blocks = [Attention(hidden_size, attn_hidden_size, r_size).to(self.device) for _ in range(self.num_categories)]
        #self.classifier_blocks = [Classifier(cls_hidden_size, r_size, 2).to(self.device) for _ in range(self.num_categories)]
        self.attention_1 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_2 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_3 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_4 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_5 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_6 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_7 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_8 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_9 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_10 = Attention(hidden_size, attn_hidden_size, r_size)
        self.attention_11 = Attention(hidden_size, attn_hidden_size, r_size)

        self.classifier_1 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_2 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_3 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_4 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_5 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_6 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_7 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_8 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_9 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_10 = AuxClassifier(cls_hidden_size, r_size, 2)
        self.classifier_11 = AuxClassifier(cls_hidden_size, r_size, 2)

        self.max_pooling = nn.MaxPool1d(hidden_size * 2)


    def forward(self, x):
        embedding_states = self.embedding(x)  # embedding_states : (batch, length, 512)
        hidden_states, _ = self.lstm(embedding_states) # hidden_states : (batch, length, lstm_hidden_states * 2)

        outputs=[]
        attn_scores=[]
        # for i in range(self.num_categories):
        #     embs, attn_score = globals()['self.attention_'+str(i)](hidden_states) # embs : (batch, r_size, lstm_hidden_states * 2)
        #     pooled = torch.squeeze(self.max_pooling(embs), 2) # pooled : (batch, r_size)
        #     output = globals()['self.classifier_'+str(i)](pooled) # outputs_1 : (batch, n_class)

        #     outputs.append(output)
        #     attn_scores.append(attn_score)

        embs1,attn_score1 = self.attention_1(hidden_states)
        pooled1 = torch.squeeze(self.max_pooling(embs1),2)
        output1 = self.classifier_1(pooled1)

        embs2,attn_score2 = self.attention_1(hidden_states)
        pooled2 = torch.squeeze(self.max_pooling(embs2),2)
        output2 = self.classifier_2(pooled2)

        embs3,attn_score3 = self.attention_3(hidden_states)
        pooled3 = torch.squeeze(self.max_pooling(embs3),2)
        output3 = self.classifier_3(pooled3)

        embs4,attn_score4 = self.attention_4(hidden_states)
        pooled4 = torch.squeeze(self.max_pooling(embs4),2)
        output4 = self.classifier_4(pooled4)

        embs5,attn_score5 = self.attention_5(hidden_states)
        pooled5 = torch.squeeze(self.max_pooling(embs5),2)
        output5 = self.classifier_5(pooled5)

        embs6,attn_score6 = self.attention_6(hidden_states)
        pooled6 = torch.squeeze(self.max_pooling(embs6),2)
        output6 = self.classifier_6(pooled6)

        embs7,attn_score7 = self.attention_7(hidden_states)
        pooled7 = torch.squeeze(self.max_pooling(embs7),2)
        output7 = self.classifier_7(pooled7)

        embs7,attn_score7 = self.attention_7(hidden_states)
        pooled7 = torch.squeeze(self.max_pooling(embs7),2)
        output7 = self.classifier_7(pooled7)

        embs8,attn_score8 = self.attention_8(hidden_states)
        pooled8 = torch.squeeze(self.max_pooling(embs8),2)
        output8 = self.classifier_8(pooled8)

        embs9,attn_score9 = self.attention_9(hidden_states)
        pooled9 = torch.squeeze(self.max_pooling(embs9),2)
        output9 = self.classifier_9(pooled9)

        embs10,attn_score10 = self.attention_10(hidden_states)
        pooled10 = torch.squeeze(self.max_pooling(embs10),2)
        output10 = self.classifier_10(pooled10)

        embs11,attn_score11 = self.attention_11(hidden_states)
        pooled11 = torch.squeeze(self.max_pooling(embs11),2)
        output11 = self.classifier_11(pooled11)

        outputs=[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11]
        attn_scores=[attn_score1,attn_score2,attn_score3,attn_score4,attn_score5,attn_score6,attn_score7,attn_score8,attn_score9,attn_score10,attn_score11]
        return outputs,attn_scores


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

        return emb_matrix, x

class AuxClassifier(nn.Module):
    def __init__(self, hidden_size, prior_size, n_class):
        super(AuxClassifier, self).__init__()

        self.ff = nn.Linear(prior_size, hidden_size)
        self.pred = nn.Linear(hidden_size, n_class)

        self.relu =nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.ff(x)) # x : (batch, ff_hidden_size)
        outputs = self.pred(x) # outputs : (batch, n_class)

        return self.softmax(outputs)

class PrmClassifier(nn.Module):
    def __init__(self, hidden_size, prior_size, n_class):
        super(PrmClassifier, self).__init__()

        self.ff = nn.Linear(prior_size, hidden_size)
        self.pred = nn.Linear(hidden_size, n_class)

        self.relu =nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.ff(x)) # x : (batch, ff_hidden_size)
        outputs = self.pred(x) # outputs : (batch, n_class)

        return self.softmax(outputs)
