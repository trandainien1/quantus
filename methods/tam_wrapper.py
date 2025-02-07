import torch

# try:
from transition_attention_maps.baselines.ViT.ViT_explanation_generator import LRP
from transition_attention_maps.baselines.ViT.ViT_new import VisionTransformer, _conv_filter, _cfg
from transition_attention_maps.baselines.ViT.helpers import load_pretrained
from timm.models.vision_transformer import default_cfgs as vit_cfgs
print('SUCCESS: tam was successfully imported.')
# except:
#     print('ERROR: tam was not found.')

def vit_base_patch16_224(pretrained=True, model_name="vit_base_patch16_224", pretrained_cfg='orig_in21k_ft_in1k', **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)

    cfg = _cfg(url=vit_cfgs[model_name].cfgs[pretrained_cfg].url, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    model.default_cfg = cfg

    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
   
    return model.cuda()

class TAMWrapper:
    def __init__(self, model, start_layer=0, steps=20, **kwargs):

        self.model = vit_base_patch16_224()
        self.model.eval()
        assert isinstance(self.model, VisionTransformer), '[ASSERT] Transformer architecture not recognised.'

        self.method = LRP(self.model)
        self.start_layer = start_layer
        self.steps = steps
        
        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', type(self.method), end='\n\n')

    def attribute(self, x, target=None):
        with torch.enable_grad():
            saliency_map = self.method.transition_attention_maps(x, index=target, start_layer=self.start_layer, steps=self.steps)
            return saliency_map.reshape(14, 14)
