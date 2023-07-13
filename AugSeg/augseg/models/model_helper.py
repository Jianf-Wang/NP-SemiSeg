import importlib
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
        if self._use_auxloss:
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        pretrained_model_url = enc_cfg["pretrain"]
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"], pretrain_model_url=pretrained_model_url)
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

    def _build_module(self, mtype, kwargs, pretrain_model_url=None):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if pretrain_model_url is None:
            return cls(**kwargs)
        else:
            return cls(pretrain_model_url=pretrain_model_url, **kwargs)

    def forward(self, x,  deterministic_memory, latent_memory, flag_use_fdrop=False,  x_context_in=None, labels_target_in=None, labels_context_in=None, forward_times=5, phase_train=True):
        h, w = x.shape[-2:]
        b = x.shape[0]
        if self._use_auxloss:
          if phase_train:
            f1, f2, feat1, feat2 = self.encoder(x)
            f1_c, f2_c, feat1_c, feat2_c = self.encoder(x_context_in) 

            outs, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, x_context_in=[f1_c, f2_c, feat1_c, feat2_c], labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
            #outs = self.decoder([f1, f2, feat1, feat2])
            pred_aux = self.auxor(feat1)

            # upsampling
            _, b, c, h_s, w_s = outs.size()
            outs = outs.view(forward_times*b, c, h_s, w_s)
            outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
            outs = outs.view(forward_times, b, c, h, w)
            pred_aux = F.interpolate(pred_aux, (h, w), mode="bilinear", align_corners=True)
            
            return outs, pred_aux, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory
          else:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x) 
                outs = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
                # upsampling
                _, b, c, h_s, w_s = outs.size()
                outs = outs.view(forward_times*b, c, h_s, w_s)
                outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
                outs = outs.view(forward_times, b, c, h, w)

                pred_aux = F.interpolate(pred_aux, (h, w), mode="bilinear", align_corners=True)

                return outs, pred_aux 
        else:
            '''
            if flag_use_fdrop:
             if phase_train:
                f1, f2, feat1, feat2 = self.encoder(x)
                f1 = nn.Dropout2d(0.5)(f1)
                feat2 = nn.Dropout2d(0.5)(feat2)
                outs, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, x_context_in=[f1_c, f2_c, feat1_c, feat2_c], labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
                _, b, c, h_s, w_s = outs.size()
                outs = outs.view(forward_times*b, c, h_s, w_s)
                outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
                outs = outs.view(forward_times, b, c, h, w)
                return outs, None, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory
             else:
                f1, f2, feat1, feat2 = self.encoder(x)
                f1 = nn.Dropout2d(0.5)(f1)
                feat2 = nn.Dropout2d(0.5)(feat2)
                outs = self.decoder([f1, f2, feat1, feat2], deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
                _, b, c, h_s, w_s = outs.size()
                outs = outs.view(forward_times*b, c, h_s, w_s)
                outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
                outs = outs.view(forward_times, b, c, h, w)
                return outs, None
            else:
            '''
            if phase_train:
                feat = self.encoder(x)
                feat_c = self.encoder(x_context_in)
                outs, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory = self.decoder(feat, deterministic_memory, latent_memory, x_context_in=feat_c, labels_target_in=labels_target_in, labels_context_in=labels_context_in, forward_times=forward_times, phase_train=phase_train)
                _, b, c, h_s, w_s = outs.size()
                outs = outs.view(forward_times*b, c, h_s, w_s)
                outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
                outs = outs.view(forward_times, b, c, h, w)
                return outs, None, mean, sigma, mean_c, sigma_c, deterministic_memory, latent_memory
            else:
                feat = self.encoder(x)
                outs = self.decoder(feat, deterministic_memory, latent_memory, forward_times=forward_times, phase_train=phase_train)
                _, b, c, h_s, w_s = outs.size()
                outs = outs.view(forward_times*b, c, h_s, w_s)
                outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
                outs = outs.view(forward_times, b, c, h, w)
                return outs, None
 
