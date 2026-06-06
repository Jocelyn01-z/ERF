import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from random import sample
from einops import rearrange


class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])# 80
        self.enc_2 = Linear(dims[0], dims[1])# 80
        self.enc_3 = Linear(dims[1], dims[2])# 1500
        self.z_layer = Linear(dims[2], n_z)#512
        self.z_b0 = nn.BatchNorm1d(n_z)
        # self.z_soft = nn.Softplus()

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        # print(enc_h3.shape)
        z = self.z_b0(self.z_layer(enc_h3))
        # z=self.z_soft(z)
        return z


class decoder1(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder1, self).__init__()
        self.dec_0 = Linear(2*n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar
class decoder2(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder2, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar



class net(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(net, self).__init__()


        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)#[80,80,1500]

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder2_list = nn.ModuleList([decoder2(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.regression = Linear(1*n_z, nLabel)
        
        self.act = nn.Sigmoid()
        self.nLabel = nLabel
        self.BN = nn.BatchNorm1d(n_z)
        self.nz=n_z
        self.num_views=len(n_input)
        
        self.classifier_view=Linear(n_z, self.num_views)#MLP(self.nz,512,self.num_views)
        
        self.fc1 = torch.nn.Linear(512, nLabel*2)
    
        
       
    def forward(self, mul_X, we,G,mode,sigma):
        
        batch_size = mul_X[0].shape[0]
        summ = 0
        prop = sigma
        share_zs = []
        share_out=[]
        mass_list=[]
        alpha_list=[]
        beta_list=[]
        evidences = dict()
        if mode =='train':
            for i,X in enumerate(mul_X):
                mask_len = int(prop*X.size(-1))
                mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(mask.shape[1])[:mask_len]
                    mask[j, zero_indices] = 0
                mul_X[i] = mul_X[i].mul(mask)

        for enc_i, enc in enumerate(self.encoder_list):        
            z_i = enc(mul_X[enc_i])
            # Evidential Deep Learning 
            out = self.fc1(z_i) 
            out = F.softplus(out)
            alpha, beta = torch.split(out, self.nLabel, 1)
            
            alpha_list.append(alpha)
            beta_list.append(beta)
            
            u_c = uncertainty(alpha, beta, self.nLabel,G)
            mass = (1 - u_c).unsqueeze(1)
            mass_list.append(mass)
            
            share_out.append(out)
            share_zs.append(z_i)
            
        M = torch.cat(mass_list, dim=1)
        M = l1_normalize_with_float_mask(M,we)
        M = M*we
        
        s_z=torch.zeros_like(share_zs[0])
        for i in range(len(share_zs)):
            specific_view_mass= M[:, i]
            result = torch.einsum('ij,i->ij',share_zs[i], specific_view_mass)
            s_z=s_z+result
        z = s_z.mul(s_z.sigmoid_()) 
        # z = F.relu(z)
        
        
        x_bar_list_c = []
        for dec_i, dec in enumerate(self.decoder2_list):
            x_bar_list_c.append(dec(z))
             
        logi = self.regression(z)
        yLable = self.act(logi)
        
        return x_bar_list_c,yLable, z, share_zs,alpha_list, beta_list
    

def uncertainty(B_alpha, B_beta, num_class,G):
    uncertainty_item = (2/(B_alpha+B_beta+2))*G#(n,c)
    uncertainty_mean = torch.sum(uncertainty_item,dim=1)/torch.sum(G,dim=1) #
    return uncertainty_mean

def l1_normalize_with_float_mask(A, mask):
    
    masked_A = torch.where(mask == 1.0, A, torch.zeros_like(A))
    l1_norm = masked_A.sum(dim=1, keepdim=True)
    l1_norm[l1_norm == 0] = 1.0  
    normalized_A = torch.where(mask == 1.0, A / l1_norm, A)
    return normalized_A
class AsymmetricBetaLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.1, k=2):
        super(AsymmetricBetaLoss, self).__init__()
        self.k = k
        self.clip = clip  
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
   
    def forward(self, B_alpha, B_beta, y):
        B_alpha = B_alpha + 1  
        B_beta = B_beta + 1

        Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
        
        m = torch.tensor(self.clip)  
        B_alpha_m = torch.max(B_alpha - m / (1 - m) * B_beta, torch.zeros_like(B_alpha))

        Ln = torch.digamma(B_alpha_m + B_beta + self.gamma_neg) - torch.digamma(B_beta)

        torch.set_grad_enabled(False)
        w_pos = torch.ones_like(B_alpha, dtype=float)
        w_neg = torch.ones_like(B_beta, dtype=float)
        for i in range(self.gamma_pos):
            w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
        for i in range(self.gamma_neg):
            w_neg *= (B_alpha_m + i) / (B_alpha_m + B_beta + i)
        torch.set_grad_enabled(True)

        pos_loss = (w_pos * torch.pow(y, self.k) * Lp)
        neg_loss = (w_neg * torch.pow(1 - y, self.k) * Ln)

        pos_loss_mean = pos_loss.mean()
        neg_loss_mean = neg_loss.mean()
        
        return pos_loss + neg_loss
    
# class AsymmetricBetaLoss(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.1, k=2, clip_min=0.01, clip_max=0.5, adjustment_step=0.01, fixed_epochs=10):
#         super(AsymmetricBetaLoss, self).__init__()
#         self.k = k
#         self.clip = clip  
#         self.clip_min = clip_min  
#         self.clip_max = clip_max  
#         self.adjustment_step = adjustment_step  
#         self.fixed_epochs = fixed_epochs  
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
        
#     def adjust_clip_based_on_loss(self, pos_loss, neg_loss):
#         """
#         根据正负样本的损失值调整 clip 值
#         """
#         if pos_loss > neg_loss:
#             # 正样本损失较大，减小 clip 值，使模型更关注正样本
#             self.clip = min(self.clip + self.adjustment_step, self.clip_max)
#             print(self.clip)
#             print("此时正损失大于负损失")
#         else:
#             # 负样本损失较大，增大 clip 值，使模型更关注负样本
#             self.clip = max(self.clip - self.adjustment_step, self.clip_min)
#             print(self.clip)
#             print("此时正损失小于于负损失")
            

#     def forward(self, B_alpha, B_beta, y):
#         # self.set_epoch(epoch)
#         B_alpha = B_alpha + 1  # 取值范围[0,1]
#         B_beta = B_beta + 1

#         # 在前 fixed_epochs 个 epoch 内固定 clip 值为 0.5
        

#         # 计算正样本损失项
#         Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
        
#         # 计算调整后的 B_alpha_m
#         m = torch.tensor(self.clip)  # 使用当前的 clip 值
#         B_alpha_m = torch.max(B_alpha - m / (1 - m) * B_beta, torch.zeros_like(B_alpha))

#         # 计算负样本损失项
#         Ln = torch.digamma(B_alpha_m + B_beta + self.gamma_neg) - torch.digamma(B_beta)

#         # 计算权重
#         torch.set_grad_enabled(False)
#         w_pos = torch.ones_like(B_alpha, dtype=float)
#         w_neg = torch.ones_like(B_beta, dtype=float)
#         for i in range(self.gamma_pos):
#             w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
#         for i in range(self.gamma_neg):
#             w_neg *= (B_alpha_m + i) / (B_alpha_m + B_beta + i)
#         torch.set_grad_enabled(True)

#         # 计算正负样本的损失
#         pos_loss = (w_pos * torch.pow(y, self.k) * Lp)
#         neg_loss = (w_neg * torch.pow(1 - y, self.k) * Ln)

#         pos_loss_mean = pos_loss.mean()
#         neg_loss_mean = neg_loss.mean()
        
        
#         return pos_loss + neg_loss    
    

def get_model(n_stacks,n_input,n_z,Nlabel,device):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,nLabel=Nlabel).to(device)
    return model