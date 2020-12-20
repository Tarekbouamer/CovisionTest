import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, batch_size, num_layers):
        super(Encoder, self).__init__()

        self.n_features = n_features
        self.hidden_size = embedding_dim*2
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rnn1 = nn.LSTM(
          input_size=self.n_features,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=self.hidden_size,
          hidden_size=self.embedding_dim,
          num_layers=self.num_layers,
          batch_first=True
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):

        x, (h, _) = self.rnn1(x)

        x, (h, _) = self.rnn2(x)

        return x, h


class Decoder(nn.Module):

    def __init__(self, seq_len, embedding_dim, n_features, batch_size, num_layers):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.hidden_size = embedding_dim*2
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.rnn1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            batch_first=True)

        self.rnn2 = nn.LSTM(
          input_size=self.embedding_dim,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_size, n_features)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):

        x = x.repeat((self.seq_len, 1, 1)).permute(1, 0, 2)

        x, (h, _) = self.rnn1(x)

        x, (h, _) = self.rnn2(x)

        x = self.output_layer(x)

        x = x.squeeze(-1)

        return x


class LSTM_RAE(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim, batch_size, num_layers):
        super(LSTM_RAE, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size

        self.encoder = Encoder(seq_len, n_features, embedding_dim, batch_size, num_layers)

        self.decoder = Decoder(seq_len, embedding_dim, n_features, batch_size, num_layers)

    def forward(self, x):

        x = x.reshape(-1, self.seq_len, self.n_features)

        _, h = self.encoder(x)
        
        x = self.decoder(h)

        x = x.reshape(self.batch_size, -1)

        return x








