import torch
import re
#import gensim
#import biovec
import numpy as np
from Bio import SeqIO
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
AA_indx = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                   'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '-': 0,'X': 0}
# MAXLEN=100
class PadSequence:
    '''Pad the sequences of different lengths, and this class is modified slightly
    from https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
    '''
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # See the __getitem__ method in Dataset class for details to create dataset
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True) 
        sequences_L = [x[0] for x in sorted_batch]
        sequences_F = [x[1] for x in sorted_batch] # 
        sequences_padded_L = pad_sequence(sequences_L, batch_first=True)
        sequences_padded_F = pad_sequence(sequences_F, batch_first=True)

        lengths = [len(x) for x in sequences_F]

        # labels = torch.Tensor([x[2] for x in sorted_batch])

        return sequences_padded_L,sequences_padded_F,lengths#, labels 


class processingSeq(Dataset):
    """
    Define the dataset class from cazy sequence data (.fasta file).
    """
    def __init__(self, seq_file):


        self.seq_L,self.seqfull= self.readSeq(seq_file) 


    def __len__(self):
        return len(self.seqfull)#len(self.seqfull)

    def __getitem__(self, index):

        seq_tensor_L = self.to_int(self.seq_L[index])
        seqfull_tensor = self.to_int(self.seqfull[index])
        return seq_tensor_L, seqfull_tensor

    def readSeq(self, seq_file):

        seqList_L = []
        seqList_full=[]

        for seq_record in SeqIO.parse(seq_file, "fasta"):
            seq=str(seq_record.seq).upper()


            seq_100 = seq[0:100]
            if len(seq_100) < 100:
                add_aa = '-' * (100 - len(seq_100))
                seq_100 = seq_100 + add_aa

            seqList_L.append(str(seq_100)) # 

            seqList_full.append(str(seq))


        return seqList_L, seqList_full




    def to_onehot(self,seq, start=0):

        vocab_size=21
        onehot = np.zeros((len(seq), vocab_size), dtype=np.int32)
        # l = min(MAXLEN, len(seq))
        l = len(seq)
        for i in range(start, start + l):
            onehot[i, AA_indx.get(seq[i - start], 0)] = 1
        # onehot[0:start, 0] = 1
        # onehot[start + l:, 0] = 1
        return torch.FloatTensor(onehot)

    def to_int(self,seq):
        vec=[]
        for i in seq:
            vec.append(AA_indx[i])
            # vec.append(AA_indx_rna[i])
        # for i in range(len(vec), MAXLEN): 
        # 	vec.append(0)	
        return torch.LongTensor(np.array(vec))
