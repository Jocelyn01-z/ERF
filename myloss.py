import torch
import torch.nn as nn
import torch.nn.functional as F
# from audtorch.metrics.functional import pearsonr

class Loss(nn.Module):
    def __init__(self, t, Nlabel,device):
        super(Loss, self).__init__()

        self.Nlabel = Nlabel
        self.t = t
        self.device = device
        self.CE = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss()
        self.criterion= nn.CrossEntropyLoss(reduction="sum")
        # self.lable_c=dep_graph#torch.mean(dep_graph,dim=1)
        
        # self.Positive=torch.exp(-dep_graph)
        # self.Negative=1-torch.exp(dep_graph-1)

    


    def wmse_loss(self,input1, target, weight, reduction='mean'):
        # ret = (torch.diag(weight).mm(target - input)) ** 2
        # ret = torch.mean(ret)
        
        ret1 = (torch.diag(weight).mm(target - input1)) ** 2
        ret1 = torch.mean(ret1)
      
        return ret1#(ret+ret1)/2 #ret,ret1#

    
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,epoch,reduction='mean'):
        try:
            assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        except AssertionError:
            debug = target_pre.data.cpu().numpy()
            print(f"断言失败，当前 epoch: {epoch}")
        # assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        target_pre_debug=target_pre.data.cpu().numpy()
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))      
        
        if reduction=='mean':
            
            return torch.sum(res)/torch.sum(inc_L_ind)  
            
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
   
    def weighted_BCE_com_loss(self,target_pre,sub_target,inc_L_ind):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))))      
       
        return res
    

    
