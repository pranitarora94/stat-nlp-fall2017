import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

hidden_dim = 100 # GTSRB as 43 classes
embedding_dim =300
p_star_dim = 2*hidden_dim + 2*embedding_dim
num_layers = 2
A = 30
class Net(nn.Module):
    def __init__(self, config, emd_data):
        super(Net, self).__init__()
        num_embeddings = config.num_emd
        self.config = config
        self.emd = nn.Embedding(num_embeddings, embedding_dim) #TODO define num_embeddings
        self.emd.weight.requires_grad = False
        self.emd.weight.data.copy_(emd_data) #TODO get embedding data
       

        self.ffnn_p_q = nn.Linear(embedding_dim, hidden_dim) #eqn(6) FFFN(p_i).FFNN(q_j) = s_i,j

        self.bilstm_q = nn.LSTM(embedding_dim, hidden_dim, num_layers =num_layers, bias = True, dropout = 0, bidirectional=True)   #eqn(9) BILSTM(q) = q_dash
        self.ffnn_q_dash = nn.Linear(2*hidden_dim,hidden_dim) # eqn(10)
        self.w_q = nn.Linear(hidden_dim,1)
        #self.ffnn_start = nn.Linear(2*hidden_dim, hidden_dim)
        #self.ffnn_end = nn.Linear(2*hidden_dim, hidden_dim)

        #LSTM (input_size,hidden_size,num_layers,bias(bool), dropout, bidirectional(bool))
        self.bilstm_pstar = nn.LSTM(p_star_dim, hidden_dim, num_layers =num_layers, bias = True, dropout = 0, bidirectional = True)   #eqn(4) p_star to p_star_dash
        self.w_a = nn.Linear(hidden_dim, 1)
        self.ffnn_ha = nn.Linear(4*hidden_dim, hidden_dim)  # eqn(2) h_a to s_a

        self.q_hidden = self.init_hidden(num_layers, hidden_dim, config.batch_size)
        self.p_hidden = self.init_hidden(num_layers, hidden_dim, config.batch_size)
        #self.conv2_drop = nn.Dropout2d()

#TODO convert ffnn applications, add relu/dropout etc
#TODO masking 
    def forward(self, p, p_lens, q, q_lens):
        L = int(p_lens.data.max())
        M = int(q_lens.data.max())
        #print("L ", str(L))
        #batch_size = len(p_lens)
        p_emd = self.emd(p) #N*L*ED
        q_emd = self.emd(q) #N*M*ED
        p_ff = F.relu(self.ffnn_p_q(p_emd)) # TODO : add dropout? #N*L*HD
        q_ff = F.relu(self.ffnn_p_q(q_emd)) # TODO : add dropout? #N*M*HD
        for i in range (p_lens.size(0)):
            for j in range(int(p_lens[i]),L):
                p_ff[i,j] = torch.zeros(hidden_dim) # TODO check this 
        for i in range (q_lens.size(0)):
            for j in range(int(q_lens[i]),M):
                q_ff[i,j] = torch.zeros(hidden_dim) # TODO check this 
        # perform dot prof
        #s_ij = p_ff:dot(q_ff) #dim won't work
        s_ij = torch.bmm(p_ff, q_ff.permute(0, 2, 1))       # N*L*M
        a_ij = F.softmax(s_ij,-1) # no need to permute
        #q_i_align = a_ij:dot(q_ff)     #TODO make sure its per i
        q_i_align = torch.bmm(a_ij,q_emd) # N*L*ED
        if torch.cuda.is_available():
            q_i_align = q_i_align.cuda(0)

        '''
        q_i_align = torch.bmm(a_ij,q_emd) # N*L*ED

        q_dash, self.q_hidden = self.bilstm_q(q_emd.permute(1,0,2), self.q_hidden)  
        q_dash = q_dash.permute(1,0,2)
        # N * M * (2HD)

        q_dash_ff = F.relu(self.ffnn_q_dash(q_dash)) # N*M*HD
        '''
        q_lens_sorted, q_ind = torch.sort(q_lens,descending=True)
        q_sorted = q_emd[q_ind]
        #print(type(p), type(q_sorted), q_sorted.size())
        q_dash_pack = torch.nn.utils.rnn.pack_padded_sequence(q_sorted.permute(1,0,2), q_lens_sorted.data.cpu().int().numpy().tolist(), )
        q_dash_sorted, self.q_hidden = self.bilstm_q(q_dash_pack, self.q_hidden)
        q_dash, q_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(q_dash_sorted)
        q_dash = q_dash.permute(1,0,2)
        q_dash_clone = q_dash
        q_dash.index_copy_(0,q_ind, q_dash_clone)


        # N * M * (2HD)

        q_dash_ff = F.relu(self.ffnn_q_dash(q_dash)) # N*M*HD
        for i in range (q_lens.size(0)):
            for j in range(int(q_lens[i]),M):
                q_dash_ff[i,j] = torch.zeros(hidden_dim) # TODO check this 

        s_j = self.w_q(q_dash_ff)   #N*M*1
        s_j = s_j.permute(0,2,1)  #N*1*M
        a_j = F.softmax(s_j, -1)  #over j (M) #N*1*M
        #a_j = a_j.unsqueeze(1)
        # convert N*1*M to N * L * 2HD . L will get by expand first need N*HD
        #print(a_j.size(1))
        a_j = a_j.expand(self.config.batch_size, L, M)
        #print(a_j.size(1))
        q_indep = torch.bmm (a_j, q_dash)       # N * (L * M) prod N * (M * 2HD) to N * L * 2HD
        if torch.cuda.is_available():
            q_indep = q_indep.cuda(0)
        #print("L is " + str(L))
        #q_indep = q_indep.expand(self.config.batch_size, L, 2*hidden_dim)
        #print(q_indep.size(1))
        #q_indep = a_j:dot(q_dash)
        assert p_emd.size(2) == embedding_dim
        assert q_i_align.size(2) == embedding_dim
        assert q_indep.size(2) == 2*hidden_dim
        p_star = [p_emd]
        p_star.append(q_i_align)
        p_star.append(q_indep)
        p_star = torch.cat(p_star, 2)
        #p_star.extend(q_i_align).extend(q_indep) # make sure final dim is N*L * (ED + ED +2HD)
        assert p_star.size(2) == p_star_dim
        # make all p_star where p_i =0 0
        # effectively use mask
        '''
        p_star_dash, self.p_hidden = self.bilstm_pstar(p_star.permute(1,0,2), self.p_hidden) #TODO init hidden
        p_star_dash = p_star_dash.permute(1,0,2)
        assert p_star_dash.size(2) == 2*hidden_dim
        '''
        if torch.cuda.is_available():
            p_star = p_star.cuda(0)
        p_lens_sorted, p_ind = torch.sort(p_lens,descending=True)
        p_star_sorted = p_star[p_ind]
        p_star_pack = torch.nn.utils.rnn.pack_padded_sequence(p_star_sorted.permute(1,0,2), p_lens_sorted.data.cpu().int().numpy().tolist())
        p_star_dash_sorted, self.p_hidden = self.bilstm_pstar(p_star_pack, self.p_hidden) #TODO init hidden
        p_star_dash, p_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(p_star_dash_sorted)
        p_star_dash = p_star_dash.permute(1,0,2)
        p_star_dash_clone = p_star_dash.clone()
        p_star_dash.index_copy_(0, p_ind, p_star_dash_clone)
        assert p_star_dash.size(2) == 2*hidden_dim
        #p_star = p_star.cpu()

        p_star_dash = p_star_dash.permute(1,0,2) # L * N * 2HD
        h_a = torch.zeros(L*A, self.config.batch_size, 4*hidden_dim) #TODO define L, A
        for j in range(0, A):
            for i in range(0, L-j):
                temp = [p_star_dash.data[i]]
                temp.append(p_star_dash.data[i+j])
                h_a[i * A + j] = torch.cat(temp, 1)
                #h_a[i*L+j].extend(p_star_dash[i+j]) #TODO check this works as intended
        # get span scores
        h_a = h_a.permute(1,0,2) # N * (L*A) * 4HD
        if torch.cuda.is_available():
            h_a = h_a.cuda(0)
        h_a = Variable(h_a)
        #print(type(h_a.data), type(self.ffnn_ha))
	h_a_ff = F.relu(self.ffnn_ha(h_a)) # N * (L*A) * HD

        s_a = self.w_a(h_a_ff)  # N * (L * A) * 1
        s_a = s_a.squeeze(2) # N* (L*A)
        s_a_mask = torch.ones(self.config.batch_size, L*A)
        for i in range (q_lens.size(0)):
            for j in range(0,L):
                for k in range(0, A):
                    if(j+k>int(p_lens[i])):
                        s_a_mask[i,j*A+k] = 0 # TODO check this
        if torch.cuda.is_available():
            s_a_mask = s_a_mask.cuda(0)
        s_a_mask = Variable(s_a_mask)
        s_a = s_a * s_a_mask
        #print(s_a.size())
        return F.log_softmax(s_a, -1)


    def init_hidden(self, num_layers, hidden_dim, batch_size):
        #zero_t = torch.zeros(num_layers * 2, batch_size, hidden_dim)
        t = torch.Tensor(num_layers * 2, batch_size, hidden_dim).uniform_(0, 1)
        if torch.cuda.is_available():
            t = t.cuda(0)
        return (Variable(t), Variable(t))
