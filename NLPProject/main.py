import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net
from torch.autograd import Variable
import torch.optim as optim
import pickle
import json
import numpy as np
import os
from itertools import ifilter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)



class Config(object):
    def __init__(self, compared=[], **kwargs):
        self.name = "RaSoR"
        self.word_emb_data_path_prefix = 'data/preprocessed_glove_with_unks.split'  # path of preprocessed word embedding data, produced by setup.py
        self.tokenized_trn_json_path = 'data/train-v1.1.tokenized.split.json'  # path of tokenized training set JSON, produced by setup.py
        self.tokenized_dev_json_path = 'data/dev-v1.1.tokenized.split.json'  # path of tokenized dev set JSON, produced by setup.py
        self.max_ans_len = 30  # maximal answer length, answers of longer length are discarded
        self.emb_dim = 300  # dimension of word embeddings
        self.ff_dim = 100
        self.batch_size = 40
        self.max_num_epochs = 150  # max number of epochs to train for
        self.num_layers = 2  # number of BiLSTM layers, where BiLSTM is applied
        self.hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
        self.vocab_size = 114885
        self.seed = np.random.random_integers(1e6, 1e9)
        self.num_known_words = 2196018 # Size of glove embedding
        self.num_emd = self.num_known_words + 100
    def __repr__(self):
        ks = sorted(k for k in self.__dict__ if k not in ['name'])
        return '\n'.join('{:<30s}{:<s}'.format(k, str(self.__dict__[k])) for k in ks)


config = Config()

#print("Loading embedding data")
#emb_data = np.load('data/glove_emb.npy') #TODO fix so that it loads into tensor

emb_data = np.zeros((config.emb_dim, config.num_emd), dtype=np.float32)
emb = torch.from_numpy(emb_data)
#print("Loading embedding indices")
with open('data/glove.840B.300d_str_idx.pickle') as f:
    embed_str_to_ind = pickle.load(f)

loss_function = nn.NLLLoss()
if torch.cuda.is_available():
    loss_function = loss_function.cuda(0)

def calc_ans_index(start,end):
    return start*config.max_ans_len + (end-start)


def convert_word_to_ind(p):
    global embed_str_to_ind
    res = []
    for p_word in p:
        if(embed_str_to_ind.get(p_word)):
            res.append(embed_str_to_ind[p_word])
        else:
            #print(p_word + " is unknown , mapped to something" )
            res.append(config.num_known_words + np.random.random_integers(0, 99)) #TODO change to random embed index
    return res


def convert_to_tensor(p):
    cols = len(p)
    rows = max(len(l) for l in p)
    m = np.zeros((cols, rows))
    for rowind, row in enumerate(p):
        m[rowind, :len(row)] = np.array(row)

    return torch.from_numpy(m).long()


def load_train_data(file_name):

    print("Loading json data")
    with open(file_name) as f:
        data = json.load(f)
    p = []
    q = []
    answers_st_end = []
    q_to_p_id = []
    p_index = -1
    for article in data['data'][:2]:
        paragraphs = article['paragraphs'][:40]
        for paragraph in paragraphs:
            p.append(convert_word_to_ind(paragraph['context_tokens']))
            p_index += 1
            for qas in paragraph['qas']:
                q.append(convert_word_to_ind(qas['question_tokens']))
                q_to_p_id.append(p_index)
                ans = []
                for answer in qas['answers']:
                    if answer['valid'] == 'true':
                        ans.append([answer['start_token_idx'], answer['end_token_idx']])
                if len(ans) == 0:
                    del q[-1]
                else:
                    answers_st_end.append(ans[0])

    answers = np.array([calc_ans_index(st,en) for st,en in answers_st_end])
    #print(type(p), type(p[0][0]))


    p = convert_to_tensor(p)
    q = convert_to_tensor(q)
    q_to_p_id = torch.LongTensor(q_to_p_id)
    p_lens = torch.LongTensor([len(pi) for pi in p])
    q_lens = torch.LongTensor([len(qi) for qi in q])
    answers = torch.FloatTensor(answers).long()
    return p, p_lens, q, q_lens, q_to_p_id, answers

comp_p, comp_p_lens, comp_q, comp_q_lens, q_to_p_id, comp_answers = load_train_data('data/train-v1.1.tokenized.split.json')
#print("comp_size = " + str(comp_q.size))

def train(epoch, model):
    global comp_p, comp_p_lens, comp_q, comp_q_lens, q_to_p_id, comp_answers
    #comp_p, comp_p_lens, comp_q, comp_q_lens, q_to_p_id, comp_answers  = load_train_data()
    num_samples = comp_q.size(0)
    #print(num_samples, type(num_samples))
    if torch.cuda.is_available():
        #print("Converting to cuda")
	comp_p = comp_p.long().cuda(0)
        comp_p_lens = comp_p_lens.long().cuda(0)
        comp_q = comp_q.long().cuda(0)
        comp_q_lens = comp_q_lens.long().cuda(0)
        q_to_p_id = q_to_p_id.long().cuda(0)
        comp_answers = comp_answers.long().cuda(0)

    losses = []
    accs = []

    np_rng = np.random.RandomState(config.seed // 2)
    idxs = np.array(range(num_samples))   #TODO check logic
    np_rng.shuffle(idxs)
    model.train()
    parameters = ifilter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum) #TODO use adam?

    all_indices = range(0, num_samples, config.batch_size)
    for batch_idx, st_in in enumerate(all_indices, 1):    #TODO why 1 ?
        batch_idxs = idxs[st_in:min(st_in+config.batch_size, num_samples)]
        qtn_idxs = torch.from_numpy(batch_idxs).long()
        if torch.cuda.is_available():
            qtn_idxs = qtn_idxs.cuda(0)
        #print(type(qtn_idxs), type(q_to_p_id))
        #print(qtn_idxs.size(0), q_to_p_id.size(0), comp_p_lens.size(0))
        if len(batch_idxs) != config.batch_size:
            continue    # LSTM may give error
        #print("qtn_idxs:")
        #print(qtn_idxs.size())
        #print(type(qtn_idxs))
        temp_index = q_to_p_id[qtn_idxs].long()
        p = comp_p[temp_index]
        print("Max length ", comp_p_lens.max())
        p_lens = comp_p_lens[temp_index]
        q = comp_q[qtn_idxs]
        q_lens = comp_q_lens[qtn_idxs]
        answer = Variable(comp_answers[qtn_idxs])
        model.zero_grad()
        #print(type(p), type(p[0][0]))
        scores = model( Variable(p, requires_grad=False), Variable(p_lens, requires_grad=False),
                   Variable(q, requires_grad=False), Variable(q_lens, requires_grad=False))
        #print("got scores")
        #print(scores.size())
        #print(scores)
        loss = loss_function(scores, answer)
        _, pred_a = torch.max(scores, 1)
        #pred_a = pred_a.squeeze(1)

        accuracy = torch.eq(pred_a, answer).float().mean()
        print(accuracy)
        print("Backprop started!")
        loss.backward()
        print("Backprop completed!")
        optimizer.step()

        losses.append(loss.data[0])
        accs.append(accuracy.data[0])
	
	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, st_in, num_samples,
                100. * st_in / num_samples, loss.data[0]))
        #if batch_idx % 500 == 0:
            #save frequency, creates a 140MB file
         #   torch.save(model.state_dict(), './model' +str(batch_idx)+ '.pth')

    trn_loss = np.average(losses)
    trn_acc = np.average(accs)
    return trn_loss, trn_acc

v_comp_p, v_comp_p_lens, v_comp_q, v_comp_q_lens, v_q_to_p_id, v_comp_answers = load_train_data('data/dev-v1.1.tokenized.split.json')
def validation(model):
    # comp_p, comp_p_lens, comp_q, comp_q_lens, q_to_p_id, comp_answers  = load_train_data()
    global v_comp_p, v_comp_p_lens, v_comp_q, v_comp_q_lens,v_q_to_p_id, v_comp_answers
    print("Validation")
    if torch.cuda.is_available():
	print("Validation cuda")
        v_comp_p = v_comp_p.long().cuda(0)
        v_comp_p_lens = v_comp_p_lens.long().cuda(0)
        v_comp_q = v_comp_q.long().cuda(0)
        v_comp_q_lens = v_comp_q_lens.long().cuda(0)
        v_q_to_p_id = v_q_to_p_id.long().cuda(0)
        v_comp_answers = v_comp_answers.long().cuda(0)

    model.eval()
    losses = []
    accs = []
    num_samples = v_comp_q.size(0)
    np_rng = np.random.RandomState(config.seed // 2)
    idxs = np.array(range(num_samples))  # TODO check logic
    np_rng.shuffle(idxs)
    # model.train()
    all_indices = range(0, num_samples, config.batch_size)
    print(num_samples)
    print(all_indices)
    for batch_idx, st_in in enumerate(all_indices, 1):  # TODO why 1 ?
        batch_idxs = idxs[st_in:min(st_in + config.batch_size, num_samples)]
        qtn_idxs = torch.from_numpy(batch_idxs).long()
        if torch.cuda.is_available():
            qtn_idxs = qtn_idxs.cuda(0)
        if len(batch_idxs) != config.batch_size:
            print(len(batch_idxs), config.batch_size)
	    continue  # LSTM may give error
	print("staring batch", st_in)
        print("Max length ", v_comp_p_lens.max())
	temp_index = v_q_to_p_id[qtn_idxs].long()
	p = v_comp_p[temp_index]
        p_lens = v_comp_p_lens[temp_index]
        q = v_comp_q[qtn_idxs]
        q_lens = v_comp_q_lens[qtn_idxs]
        #answer = comp_answers[qtn_idxs]
	answer = Variable(v_comp_answers[qtn_idxs])
        scores = model(Variable(p, requires_grad=False), Variable(p_lens, requires_grad=False),
                       Variable(q, requires_grad=False), Variable(q_lens, requires_grad=False))
	print("calc val scores")
        loss = loss_function(scores, answer)
        _, pred_a = torch.max(scores, 1)
        #pred_a = pred_a.squeeze(1)
        accuracy = torch.eq(pred_a, answer).float().mean()

        losses.append(loss.data[0])
        accs.append(accuracy.data[0])

        if batch_idx % 20 == 0:
            print("Dev loss: {} accuracy:{} batchID:{}".format(loss.data[0], accuracy.data[0], batch_idx))

    dev_loss = np.average(losses)
    dev_acc = np.average(accs)
    return dev_loss, dev_acc




def main():
    print("Initialising model")
    model = Net(config, emb)
    print("Model initialised")
    if torch.cuda.is_available():
        model = model.cuda()

    #load older model if it exists
    if os.path.isfile('./squad_model.pth'):
        model.load_state_dict(torch.load('./squad_model.pth'))

    for epoch in range(args.epochs):
        print("Starting epoch " + str(epoch+1))
        trn_loss, trn_accuracy = train(epoch, model)
	print("trn_accuarcy: ",str(trn_accuracy))
	print("trn_loss: ",str(trn_loss))
        dev_loss, dev_accuracy = validation(model)
	print("dev_accuarcy: ",str(dev_accuracy))
        print("dev_loss: ",str(dev_loss))
        if (epoch%5==4):
            model_file = 'model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
            print( '\nSaved model to ' + model_file + '.')


main()
