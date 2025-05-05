import torch
from mmcv.runner import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .GaussianLSS_encoder import GaussianLSSTransformerEncoder
from .GaussianLSS_decoder import GaussianLSSTransformerDecoder

@TRANSFORMER.register_module()
class GaussianLSSTransformer(BaseModule):
    def __init__(self, embed_dims, encoder, decoder, init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(GaussianLSSTransformer, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.encoder = GaussianLSSTransformerEncoder(**encoder)
        self.decoder = GaussianLSSTransformerDecoder(**decoder)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, attn_mask, img_metas):
        bev = self.encoder(mlvl_feats, img_metas)
        bev = bev.flatten(2).permute(0, 2, 1) # b d h w -> b (h w) d
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, bev, attn_mask, img_metas)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds