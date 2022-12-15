
import os
import torch
import numpy as np
import pandas as pd
import argparse
from Bio import SeqIO
import random
import torch.nn.functional as F
from dataGenerate import processingSeq, PadSequence
from torch.utils.data import DataLoader
from model import Digerati

import warnings
warnings.filterwarnings("ignore")
# os.environ['CUDA_VISIBLE_DEVICES'] ='2'
root_path = os.getcwd()+'/Digerati'

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input_file", default='train.fasta', help="data path to the training data")
parser.add_argument('-o', "--output_file", default='test.fasta', help="data path to the test data")
args = parser.parse_args()




def readSeqIFor(seq_file):

    seqlist = []
    seqIDList = []
    for seq_record in SeqIO.parse(seq_file, "fasta"):
        seq=str(seq_record.seq).upper()
        seq_id=str(seq_record.description)
        seqlist.append(seq)
        seqIDList.append(seq_id)
    return  seqlist,seqIDList


def predict_func(outputs):
    preds_mapper={'0':'Non-PE_PGRS/PPE','1':'PPE','2':'PE_PGRS'}
    outputs = np.argmax(outputs, axis=1)
    return np.vectorize(preds_mapper.get)(outputs.astype(str))
    #取最大

def evaluation(model,device,testIter):

    # load checkpoint of best model to do evaluation
    checkpoint = torch.load(root_path+'/model/model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    print('\n')
    print('*'*21 + 'Model testing' + '*'*21)
    model.eval()
    testresults=[]
    # y_true_class=[]
    with torch.no_grad():
        for index,data in enumerate(testIter):

            seq_L,seqfull, _,= data
            seq_L,seqfull,= seq_L.to(device),seqfull.to(device)

            output = model(seq_L,seqfull)

            output = F.softmax(output, dim=1)
            #output = torch.sigmoid(output)
            # get the predicted labels
            if output.device == 'cpu':
                y_pred = output.detach().numpy()
                
            else:
                y_pred = output.cpu().detach().numpy()

            testresults.append(y_pred)

    res= np.vstack(testresults)
    y_pred_class=predict_func(res)
    
    seqList,seqIDlist=readSeqIFor(args.input_file)
    resultDF=pd.DataFrame()
    resultDF['seqID']=seqIDlist
    # resultDF['sequence']=seqList
    resultDF['label']=y_pred_class
    # list = list(res[0][1])
    pro_list=[max(list(l)) for l in res]
    # pro_list = list(pro_list)
    # for i in pro_list:
    #     resultDF['probo']=pro_list[i]
    # resultDF['probo_1']=list(res[:,1])
    resultDF['probobility']=pro_list
    # print(resultDF,'/n',pro_list,type(pro_list))
    # print(type(resultDF))
    resultDF.to_csv(args.output_file, header=None )
# index=False

    
def main():
    # configure the device
    device = torch.device('cpu')
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_data = processingSeq(args.input_file)
    testIter = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=PadSequence(), num_workers=2)

    model = Digerati().to(device)
    evaluation(model,device,testIter)



if __name__ == "__main__":
    main()
