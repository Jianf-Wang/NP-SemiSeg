import torch
import torch.nn as nn
import torch.nn.functional as F
 
import math
import numpy as np
from torchvision.transforms import RandomCrop





class MLP(nn.Module):
    def __init__(self, layer_sizes=[512, 512], last_act=False):

        super(MLP, self).__init__()

        self.MLP = nn.Sequential()
        
        if last_act:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU(inplace=True))
        else:
         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
            if i < (len(layer_sizes[:-1])-1):
               self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU(inplace=True))
        
        

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)

        
    def forward(self, x):
        x = self.MLP(x)
        return x
        
  

class Conv_1(nn.Module):
    def __init__(self, input_dim, latent_dim, last_act=False):

        super(Conv_1, self).__init__()
        

        if last_act:

          self.Conv_1= nn.Sequential(
            nn.Conv2d(input_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(latent_dim), 
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
          )
        else:
          self.Conv_1= nn.Sequential(
            nn.Conv2d(input_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0, bias=True),
          )
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)
      
    def forward(self, x):
        x = self.Conv_1(x)
        return x

class Conv_Decoder(nn.Module):
    def __init__(self, input_dim, num_classes):

        super(Conv_Decoder, self).__init__()
       
        # Decoder with classification layer 

        self.Conv_Decoder = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, stride=1, padding=1, bias=True), 
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256,  kernel_size=3, stride=1, padding=1, bias=True), 
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        ) 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                 nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.Conv_Decoder(x)
        return x


class NP_HEAD(nn.Module):
    def __init__(self, input_dim, latent_dim=32, num_classes=21, memory_max_length=2560):
        super(NP_HEAD, self).__init__()
 
        self.memory_dim_transform = Conv_1(input_dim=input_dim, latent_dim=latent_dim, last_act=True) 
        self.mean_logvar_net = MLP(layer_sizes=[latent_dim, latent_dim, latent_dim* 2])

        self.num_classes = num_classes
        self.decoder =  Conv_Decoder(input_dim= (2 * latent_dim + input_dim), num_classes=num_classes )
 
        self.memory_max_length = memory_max_length 
        self.latent_dim = latent_dim
        self.input_dim = input_dim 
 
 
    def reparameterize(self, mean, std): 
         # (2, latent_dim, 193, 193)

        eps = torch.randn_like(std)
        return eps * std + mean
 

    def forward(self, x_target_in, deterministic_memory, latent_memory, x_context_in=None, labels_target_in=None, labels_context_in=None, forward_times=5, phase_train=True):

        # x_target_in: torch.Size([2, 512, 193, 193])
        # label_target_in: torch.Size([2, 1, 193, 193])
  

        B, D, W, H = x_target_in.size() 


        if phase_train:
          if labels_target_in.dim() == 3:
             labels_target_in = labels_target_in.unsqueeze(1)
          if labels_context_in.dim() == 3:
             labels_context_in = labels_context_in.unsqueeze(1)
 
          sizes = labels_target_in.size()
          w = sizes[-1]
          h = sizes[-1]
 

          x_target_in_resize = F.interpolate(
                x_target_in, size=(w, h), mode="bilinear", align_corners=True
            )
          x_context_in_resize = F.interpolate(
                x_context_in, size=(w, h), mode="bilinear", align_corners=True
            )  

          with torch.no_grad():
           x_context_in_deterministic = self.memory_dim_transform(x_context_in_resize)
           x_target_in_latent = self.memory_dim_transform(x_target_in_resize)

          
           for i in range(self.num_classes): 
            mask_target = labels_target_in.eq(i) 

            x_target_in_latent_select = torch.masked_select(x_target_in_latent, mask_target).view(-1, self.latent_dim)
            latent_memory[i] = torch.cat((latent_memory[i], x_target_in_latent_select.detach()), dim=0)
            if latent_memory[i].size(0) > self.memory_max_length:
                   Diff = latent_memory[i].size(0) -  self.memory_max_length
                   latent_memory[i] = latent_memory[i][Diff:, :]   


            mask_context = labels_context_in.eq(i)
            x_context_in_deterministic_select = torch.masked_select(x_context_in_deterministic, mask_context).view(-1, self.latent_dim)
            deterministic_memory[i] = torch.cat((deterministic_memory[i], x_context_in_deterministic_select.detach()), dim=0)
            if deterministic_memory[i].size(0) > self.memory_max_length:
                   Diff = deterministic_memory[i].size(0) -  self.memory_max_length
                   deterministic_memory[i] = deterministic_memory[i][Diff:, :]     

           temporal_latent = []
           temporal_deterministic = []
           for i in range(len(latent_memory)):
              temporal_latent.append(latent_memory[i].mean(0))
              temporal_deterministic.append(deterministic_memory[i].mean(0))
           latent_memory_centers = torch.stack(temporal_latent)    
           deterministic_memory_centers = torch.stack(temporal_deterministic)  
 
        else:
          latent_memory_centers = latent_memory
          deterministic_memory_centers = deterministic_memory


        # (2, 21, 512, 193, 193)  
        latent_memory_centers_expand = latent_memory_centers.unsqueeze(2).unsqueeze(3).unsqueeze(0).expand(B, -1, -1, W, H)
        deterministic_memory_centers_expand = deterministic_memory_centers.unsqueeze(2).unsqueeze(3).unsqueeze(0).expand(B, -1, -1, W, H)

        x_target_in_latent_origin = self.memory_dim_transform(x_target_in.detach())

        target_residual = x_target_in_latent_origin.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1) - latent_memory_centers_expand
        context_residual = x_target_in_latent_origin.unsqueeze(1).expand(-1, self.num_classes, -1, -1, -1) - deterministic_memory_centers_expand

        target_residual_square = torch.square(target_residual)
        context_residual_square = torch.square(context_residual)

        sim_target =  -1.0 * target_residual_square 
        sim_context = -1.0 * context_residual_square 
  
          # (2, 21, 512, 193, 193)
        target_attention = torch.softmax(sim_target, dim=1)
        context_attention = torch.softmax(sim_context, dim=1)

        # (2, 512, 193, 193)
        target_accumulate = torch.sum(target_attention * latent_memory_centers_expand, dim=1)
        context_accumulate = torch.sum(context_attention * deterministic_memory_centers_expand, dim=1)

        # (2, latent_dim, 193, 193)
        deterministic_context = context_accumulate
        mean_logvar = self.mean_logvar_net(torch.mean(target_accumulate, dim=(2, 3)))
        mean_logvar_context = self.mean_logvar_net(torch.mean(context_accumulate, dim=(2, 3)))


        mean_all = mean_logvar[:, :self.latent_dim]
        log_var = mean_logvar[:, self.latent_dim:]
        sigma_all = 0.1 + 0.9 * F.softplus(log_var) 

        mean_c_all = mean_logvar_context[:, :self.latent_dim]
        log_var_c = mean_logvar_context[:, self.latent_dim:]
        sigma_c_all = 0.1 + 0.9 * F.softplus(log_var_c) 

        # (forward_times, 2, latent_dim, 193, 193)
        for i in range(0, forward_times):
          z = self.reparameterize(mean_all, sigma_all)
          z = z.unsqueeze(0)
          if i == 0:
              latent_z_target = z
          else:
              latent_z_target = torch.cat((latent_z_target, z))

         
        deterministic_context = torch.mean(deterministic_context, dim=(2, 3))
 
        #x_target_in: torch.Size([forward_times, 2, 512, 193, 193])
        x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times ,-1, -1, -1, -1)
        # (forward_times, 2, latent_dim, 193, 193) 
        context_representation_deterministic_expand = deterministic_context.unsqueeze(0).unsqueeze(3).unsqueeze(4).expand(forward_times, -1, -1,  W, H)
        latent_z_target_expand = latent_z_target.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, W, H) 
        decoder_input_cat = torch.cat((latent_z_target_expand, x_target_in_expand, context_representation_deterministic_expand), dim=2)

        ################## decoder ##################
        decoder_input_cat_view = decoder_input_cat.view(forward_times * B, -1, W, H)
        output_view = self.decoder(decoder_input_cat_view) 
        output = output_view.view(forward_times, B, -1, W, H)

        if phase_train:
           return output, mean_all, sigma_all, mean_c_all, sigma_c_all, deterministic_memory, latent_memory
        else:
           return output
