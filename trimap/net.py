import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
    
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False, v_mask=True, q_mask=True, attn=None):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(att_maps.size())
            att_maps.data.masked_fill_(mask.data, -float('inf'))
        if q_mask:
            mask = (0 == q.abs().sum(2)).unsqueeze(1).unsqueeze(2).expand(att_maps.size())
            att_maps.data.masked_fill_(mask.data, -float('inf'))
        if softmax:
            p = nn.functional.softmax(att_maps.reshape(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

    
class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.2):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MLPDecoder(nn.Module):
    '''
    MLP decoder for TCR chain
    '''
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.fc1(x)))
        x = self.leakyrelu(self.bn2(self.fc2(x)))
        x = self.leakyrelu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

def kl_div(mu, log_var):
    """
    KL divergence between the prior and posterior distributions
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    
class classifier(nn.Module):
    """
    Classifier for TCR chain
    """
    def __init__(self, input_size):
        super(classifier, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        
    def forward(self, z):
        x = self.seq(z)
        return x

class Conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool='max', drop=0.2):
        super(Conv_block, self).__init__()
    
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class Deconv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool='max', drop=0.2):
        super(Deconv_block, self).__init__()
    
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class Encoder(nn.Module):
    """Convolution for TCR chain. """
    def __init__(self,input_size,hidden,filters=[64,128,256],kernel_sizes=[3,3,3],drop=0.2):
        super(Encoder, self).__init__()

        input_filters, length = input_size[0], input_size[1]
        
        self.cnn1 = Conv_block(input_filters, filters[0], kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]-1)//2, drop=drop)
        self.cnn2 = Conv_block(filters[0], filters[1], kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]-1)//2, drop=drop)
        self.cnn3 = Conv_block(filters[1], filters[2], kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]-1)//2, drop=drop)
            
        self.seq = nn.Sequential(
            nn.Linear(filters[-1]*length, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(drop),
            nn.Linear(512, hidden),
        )
            
    def forward(self,x):
        # CNN part
        x = self.cnn1(x)
        x = self.cnn2(x)
        xx = self.cnn3(x)
        x = xx.flatten(start_dim=1)
        x = self.seq(x)
        return x, xx
    
class Decoder(nn.Module):
    def __init__(self,hidden,output_size,filters=[256,128,64],kernel_sizes=[3,3,3,3],drop=0.2):
        super(Decoder, self).__init__()
        
        output_filters, length = output_size[0], output_size[1]
        
        self.t_linear = nn.Linear(hidden, 32*length) 
        self.deconv1 = Deconv_block(32, filters[0], kernel_size=kernel_sizes[0], padding=1, drop=drop)
        self.deconv2 = Deconv_block(filters[0], filters[1], kernel_size=kernel_sizes[1], padding=1, drop=drop)
        self.deconv3 = Deconv_block(filters[1], filters[2], kernel_size=kernel_sizes[2], padding=1, drop=drop)
        self.deconv4 = nn.ConvTranspose1d(filters[2], output_filters, kernel_size=kernel_sizes[3], padding=1)
        
    def forward(self,x):
        x = self.t_linear(x)
        x = x.view(x.shape[0],32,x.shape[1]//32)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
    
class Linear_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden, drop=0.2):
        super(Linear_block, self).__init__()
    
        self.linear1 = nn.Linear(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.mean = nn.Linear(hidden,hidden)
        self.logvar = nn.Linear(hidden,hidden)
        self.linear2 = nn.Linear(hidden,out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.linear1(x))))
        z_mean = self.mean(x)
        z_logvar = self.logvar(x)
        if self.training:
            z = self.reparameterize(z_mean, z_logvar)
            x = self.dropout(self.relu(self.bn2(self.linear2(z))))
            return z, z_mean, z_logvar, x
        else:
            x = self.relu(self.bn2(self.linear2(z_mean)))
            return z_mean, z_logvar, x