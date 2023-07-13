import importlib
import torch
import torch.nn as nn
from torch.nn import functional as F

from .decoder import Aux_Module


class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder = self._build_encoder(net_cfg["encoder"])
        self.decoder = self._build_decoder(net_cfg["decoder"])

        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),
                "sync_bn": self._sync_bn,
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x, deterministic_memory, latent_memory, x_context_in=None, labels_target_in=None, labels_context_in=None, forward_times=5, phase_train=True):
        if self._use_auxloss: 
          if phase_train:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x) 
                f1_c, f2_c, feat1_c, feat2_c = self.encoder(x_context_in) 
                 
                outs = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, x_context_in=[f1_c, f2_c, feat1_c, feat2_c], labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
            else:
                feat1, feat2 = self.encoder(x)
                feat1_c, feat2_c = self.encoder(x_context_in)
                outs = self.decoder(feat2, deterministic_memory, latent_memory, x_context_in=feat2_c, labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
          else:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x) 
                outs = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
            else:
                feat1, feat2 = self.encoder(x) 
                outs = self.decoder(feat2, deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
                 
          pred_aux = self.auxor(feat1)
          outs.update({"aux": pred_aux})
          return outs
        else: 
            
          if phase_train:
                feat = self.encoder(x) 
                feat_c = self.encoder(x_context_in)
                outs = self.decoder(feat, deterministic_memory, latent_memory, x_context_in=feat_c, labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
          else:
                feat = self.encoder(x)
                outs = self.decoder(feat, deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
 
          return outs
