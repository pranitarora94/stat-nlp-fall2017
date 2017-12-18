import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

hidden_dim = 100  # GTSRB as 43 classes
embedding_dim = 300
p_star_dim = 2 * hidden_dim + 2 * embedding_dim
num_layers = 2
A = 30


class Net(nn.Module):
    def __init__(self, config, emd_data):
        super(Net, self).__init__()
        num_embeddings = config.num_emd
        self.config = config
        self.emd = nn.Embedding(num_embeddings, embedding_dim)  # TODO define num_embeddings
        self.emd.weight.requires_grad = False
        self.emd.weight.data.copy_(emd_data)  # TODO get embedding data

        self.ffnn_p_q = nn.Linear(embedding_dim, hidden_dim)  # eqn(6) FFFN(p_i).FFNN(q_j) = s_i,j

        self.bilstm_q = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bias=True, dropout=0,
                                bidirectional=True)  # eqn(9) BILSTM(q) = q_dash
        self.ffnn_q_dash = nn.Linear(2 * hidden_dim, hidden_dim)  # eqn(10)
        self.w_q = nn.Linear(hidden_dim, 1)
        self.ffnn_start = nn.Linear(2*hidden_dim, hidden_dim)
        self.ffnn_end = nn.Linear(2*hidden_dim, hidden_dim)

        # LSTM (input_size,hidden_size,num_layers,bias(bool), dropout, bidirectional(bool))
        self.bilstm_pstar = nn.LSTM(p_star_dim, hidden_dim, num_layers=num_layers, bias=True, dropout=0,
                                    bidirectional=True)  # eqn(4) p_star to p_star_dash
        self.w_a = nn.Linear(hidden_dim, 1)
        self.ffnn_ha = nn.Linear(4 * hidden_dim, hidden_dim)  # eqn(2) h_a to s_a

        # self.q_hidden = self.init_hidden(num_layers, hidden_dim, config.batch_size)
        # self.p_hidden = self.init_hidden(num_layers, hidden_dim, config.batch_size)
        # self.conv2_drop = nn.Dropout2d()

    # TODO convert ffnn applications, add relu/dropout etc
    # TODO masking
    def forward(self, p, p_lens, q, q_lens, p_hidden, q_hidden):
        L = int(p_lens.data.max())
        M = int(q_lens.data.max())
        # print("L ", str(L))
        # batch_size = len(p_lens)
        p_emd = self.emd(p)  # N*L*ED
        q_emd = self.emd(q)  # N*M*ED
        p_ff = F.relu(self.ffnn_p_q(p_emd))  # TODO : add dropout? #N*L*HD
        q_ff = F.relu(self.ffnn_p_q(q_emd))  # TODO : add dropout? #N*M*HD

        for i in range(p_lens.size(0)):
            for j in range(int(p_lens[i]), L):
                p_ff[i, j] = torch.zeros(hidden_dim)  # TODO check this
        for i in range(q_lens.size(0)):
            for j in range(int(q_lens[i]), M):
                q_ff[i, j] = torch.zeros(hidden_dim)  # TODO check this
        # perform dot prof
        # s_ij = p_ff:dot(q_ff) #dim won't work

        s_ij = torch.bmm(p_ff, q_ff.permute(0, 2, 1))  # N*L*M
        a_ij = F.softmax(s_ij, -1)  # no need to permute
        # q_i_align = a_ij:dot(q_ff)     #TODO make sure its per i


        q_i_align = torch.bmm(a_ij, q_emd)  # N*L*ED
        if torch.cuda.is_available():
            q_i_align = q_i_align.cuda(0)

        '''
        q_i_align = torch.bmm(a_ij,q_emd) # N*L*ED

        q_dash, self.q_hidden = self.bilstm_q(q_emd.permute(1,0,2), self.q_hidden)  
        q_dash = q_dash.permute(1,0,2)
        # N * M * (2HD)

        q_dash_ff = F.relu(self.ffnn_q_dash(q_dash)) # N*M*HD
        '''
        q_lens_sorted, q_ind = torch.sort(q_lens, descending=True)
        q_sorted = q_emd[q_ind]
        # print(type(p), type(q_sorted), q_sorted.size())
        q_dash_pack = torch.nn.utils.rnn.pack_padded_sequence(q_sorted.permute(1, 0, 2),
                                                              q_lens_sorted.data.cpu().int().numpy().tolist())

        # self.q_hidden[0].detach_()
        # self.q_hidden[1].detach_()


        q_dash_sorted, q_hidden = self.bilstm_q(q_dash_pack, (q_hidden))
        q_dash, q_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(q_dash_sorted)
        q_dash = q_dash.permute(1, 0, 2)
        q_dash_clone = q_dash
        q_dash.index_copy_(0, q_ind, q_dash_clone)

        # N * M * (2HD)

        q_dash_ff = F.relu(self.ffnn_q_dash(q_dash))  # N*M*HD
        for i in range(q_lens.size(0)):
            for j in range(int(q_lens[i]), M):
                q_dash_ff[i, j] = torch.zeros(hidden_dim)  # TODO check this

        s_j = self.w_q(q_dash_ff)  # N*M*1
        s_j = s_j.permute(0, 2, 1)  # N*1*M
        a_j = F.softmax(s_j, -1)  # over j (M) #N*1*M
        # a_j = a_j.unsqueeze(1)
        # convert N*1*M to N * L * 2HD . L will get by expand first need N*HD
        # print(a_j.size(1))
        a_j = a_j.expand(self.config.batch_size, L, M)
        # print(a_j.size(1))
        q_indep = torch.bmm(a_j, q_dash)  # N * (L * M) prod N * (M * 2HD) to N * L * 2HD
        if torch.cuda.is_available():
            q_indep = q_indep.cuda(0)
        # print("L is " + str(L))
        # q_indep = q_indep.expand(self.config.batch_size, L, 2*hidden_dim)
        # print(q_indep.size(1))
        # q_indep = a_j:dot(q_dash)
        assert p_emd.size(2) == embedding_dim
        assert q_i_align.size(2) == embedding_dim
        assert q_indep.size(2) == 2 * hidden_dim
        p_star = [p_emd]
        p_star.append(q_i_align)
        p_star.append(q_indep)
        p_star = torch.cat(p_star, 2)
        # p_star.extend(q_i_align).extend(q_indep) # make sure final dim is N*L * (ED + ED +2HD)
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
        p_lens_sorted, p_ind = torch.sort(p_lens, descending=True)
        p_star_sorted = p_star[p_ind]
        p_star_pack = torch.nn.utils.rnn.pack_padded_sequence(p_star_sorted.permute(1, 0, 2),
                                                              p_lens_sorted.data.cpu().int().numpy().tolist())





        #self.p_hidden[0].detach_()
        #self.p_hidden[1].detach_()
        p_star_dash_sorted, p_hidden = self.bilstm_pstar(p_star_pack, p_hidden)  # TODO init hidden
        p_star_dash, p_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(p_star_dash_sorted)
        p_star_dash = p_star_dash.permute(1, 0, 2)
        p_star_dash_clone = p_star_dash.clone()
        p_star_dash.index_copy_(0, p_ind, p_star_dash_clone)
        assert p_star_dash.size(2) == 2 * hidden_dim
        # p_star = p_star.cpu()

        p_star_dash = p_star_dash.permute(1, 0, 2)  # L * N * 2HD
        # get span scores

        p_stt_lin = F.relu(self.ffnn_start(p_star_dash))  # (max_p_len, batch_size, ff_dim)
        p_end_lin = F.relu(self.ffnn_end(p_star_dash) ) # (max_p_len, batch_size, ff_dim)

        # (batch_size, max_p_len*max_ans_len, ff_dim), (batch_size, max_p_len*max_ans_len)
        span_lin_reshaped, span_masks_reshaped = self._span_sums(p_lens, p_stt_lin, p_end_lin, L,
                                                                 self.config.batch_size,hidden_dim, A)
        # FFNN(h_a) also contains a relu so we are apply that to the whole span_sum
        span_ff_reshaped = F.relu(span_lin_reshaped)  # (batch_size, max_p_len*max_ans_len, ff_dim)

        span_scores_reshaped = self.w_a(span_ff_reshaped)  # (batch_size, max_p_len*max_ans_len)
        span_scores_reshaped = span_scores_reshaped.squeeze()
        final_span_scores = span_masks_reshaped * span_scores_reshaped


        return F.log_softmax(final_span_scores, -1)

    def init_hidden(self, num_layers, hidden_dim, batch_size):
        t = torch.zeros(num_layers * 2, batch_size, hidden_dim)
        #t = torch.Tensor(num_layers * 2, batch_size, hidden_dim).uniform_(0, 1)
        if torch.cuda.is_available():
            t = t.cuda(0)
        return (Variable(t), Variable(t))

    def _span_sums(self, p_lens, stt, end, max_p_len, batch_size, dim, max_ans_len):
        # stt 		(max_p_len, batch_size, dim)
        # end 		(max_p_len, batch_size, dim)
        # p_lens 	(batch_size,)

        max_ans_len_range = torch.from_numpy(np.arange(max_ans_len))
        max_ans_len_range = max_ans_len_range.unsqueeze(
            0)  # (1, max_ans_len) is a vector like [0,1,2,3,4....,max_ans_len-1]
        offsets = torch.from_numpy(np.arange(max_p_len))
        offsets = offsets.unsqueeze(0)  # (1, max_p_len) is a vector like (0,1,2,3,4....max_p_len-1)
        offsets = offsets.transpose(0, 1)  # (max_p_len, 1) is row vector now like [0/1/2/3...max_p_len-1]

        end_idxs = max_ans_len_range.expand(offsets.size(0), max_ans_len_range.size(1)) + offsets.expand(
            offsets.size(0), max_ans_len_range.size(1))
        # pdb.set_trace()
        end_idxs_flat = end_idxs.view(-1, 1).squeeze(1)  # (max_p_len*max_ans_len, )
        # note: this is not modeled as tensor of size (SZ, 1) but vector of SZ size
        zero_t = torch.zeros(max_ans_len - 1, batch_size, dim)
        if torch.cuda.is_available():
            zero_t = zero_t.cuda(0)
            end_idxs_flat = end_idxs_flat.cuda(0)

        end_padded = torch.cat((end, Variable(zero_t)), 0)
        end_structed = end_padded[end_idxs_flat]  # (max_p_len*max_ans_len, batch_size, dim)
        end_structed = end_structed.view(max_p_len, max_ans_len, batch_size, dim)
        stt_shuffled = stt.unsqueeze(1)  # stt (max_p_len, 1, batch_size, dim)

        # since the FFNN(h_a) * W we expand h_a as [p_start, p_end]*[w_1 w_2] so this reduces to p_start*w_1 + p_end*w_2
        # now we can reuse the operations, we compute only once
        span_sums = stt_shuffled.expand(max_p_len, max_ans_len, batch_size,
                                        dim) + end_structed  # (max_p_len, max_ans_len, batch_size, dim)

        span_sums_reshapped = span_sums.permute(2, 0, 1, 3).contiguous().view(batch_size, max_ans_len * max_p_len, dim)

        p_lens_shuffled = p_lens.unsqueeze(1)
        end_idxs_flat_shuffled = end_idxs_flat.unsqueeze(0)

        span_masks_reshaped = Variable(end_idxs_flat_shuffled.expand(p_lens_shuffled.size(0),
                                                                     end_idxs_flat_shuffled.size(
                                                                         1))) < p_lens_shuffled.expand(
            p_lens_shuffled.size(0), end_idxs_flat_shuffled.size(1))
        span_masks_reshaped = span_masks_reshaped.float()

        return span_sums_reshapped, span_masks_reshaped
