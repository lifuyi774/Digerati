import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Digerati(nn.Module):
    ''' Define the model using CNN and LSTM '''
    def __init__(self): 
        super(Digerati, self).__init__()
        vocab_size=21
        embedding_dim=50
        self.Embedding=nn.Embedding(vocab_size, embedding_dim,padding_idx=0)

        self.kernel_size_1 = 2 
        self.pool_size_1 = 2

        self.convPool1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 30, self.kernel_size_1), nn.ReLU(True), nn.MaxPool1d(self.pool_size_1, self.pool_size_1)
        )
        self.convPool1_ = nn.Sequential(
            nn.Conv1d(embedding_dim, 30, self.kernel_size_1), nn.ReLU(True), nn.MaxPool1d(self.pool_size_1, self.pool_size_1)
        )

        self.drop1 = nn.Dropout(0.2)

        self.kernel_size_2 = 2
        self.pool_size_2 = 2
        self.convPool2 = nn.Sequential(
            nn.Conv1d(30, 30, self.kernel_size_2), nn.ReLU(True), nn.MaxPool1d(self.pool_size_2, self.pool_size_2)
        )
        self.convPool2_ = nn.Sequential(
            nn.Conv1d(30, 30, self.kernel_size_2), nn.ReLU(True), nn.MaxPool1d(self.pool_size_2, self.pool_size_2)
        )

        self.drop2 = nn.Dropout(0.2)



        # Bi-LSTM layer definition
        self.BiLstm = nn.LSTM(
            input_size = 30,    # input size
            hidden_size = 30,   # hidden size of lstm
            num_layers = 1,     # num of layers
            batch_first = True,
            bidirectional = True  # two directions of LSTM
        )

        self.BiLstm_ = nn.LSTM(
            input_size = 30,    # input size
            hidden_size = 30,   # hidden size of lstm
            num_layers = 1,     # num of layers
            batch_first = True,
            bidirectional = True  # two directions of LSTM
        )


        
        self.lstmdrop = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(2 *30, 32)  
        self.fc1_ = nn.Linear(2 *30, 32) 

        self.wq = nn.Linear(32,32)
        self.wk = nn.Linear(32,32)
        self.wv = nn.Linear(32,32)

        self.wq_ = nn.Linear(32,32)
        self.wk_ = nn.Linear(32,32)
        self.wv_ = nn.Linear(32,32)
        

        
        self.fc2 = nn.Linear(64, 3)       

        # self.criterion= nn.CrossEntropyLoss()
        # self.optimier = optim.Adam(self.parameters(), lr=learning_rate)   # set optimizer for optimization
        # self.optimier = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    def forward(self, x,x1):
        # embedding
        x=self.Embedding(x)
        x = x.permute(0, 2, 1)
        x = self.convPool1(x)
        x = self.drop1(x)
        x = self.convPool2(x)
        
        x = self.drop2(x)
        x = x.permute(0, 2 ,1)  
        _, (x, _) = self.BiLstm(x)  
        x = x.permute(1, 0 ,2)
        x = self.lstmdrop(x)
        x = x.reshape(x.size(0), -1)        
        x = self.fc1(x)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        x = torch.mm(torch.softmax(torch.mm(q,k.t())/5.65,dim=-1),v)#sqrt(32)=5.65

        
        
        x1=self.Embedding(x1)
        
        x1 = x1.permute(0, 2, 1)
        x1 = self.convPool1_(x1)
        x1 = self.drop1(x1)
        x1 = self.convPool2_(x1)
        
        
        x1 = self.drop2(x1)
        x1 = x1.permute(0, 2 ,1)  
        _, (x1, _) = self.BiLstm_(x1) 
        x1 = x1.permute(1, 0 ,2)
        

        
        x1 = self.lstmdrop(x1)
        x1 = x1.reshape(x1.size(0), -1)

        
        x1 = self.fc1_(x1)


        q1 = self.wq_(x1)
        k1 = self.wk_(x1)
        v1 = self.wv_(x1)
        x1 = torch.mm(torch.softmax(torch.mm(q1,k1.t())/5.65,dim=-1),v1)


        cat_x= torch.cat((x, x1), 1) 

        out = self.fc2(cat_x)
        return out

        
