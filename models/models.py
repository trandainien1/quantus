import timm
import AGC_methods.AGCAM.ViT_for_AGCAM as ViT_Ours
import torch.utils.model_zoo as model_zoo
MODEL = 'vit_base_patch16_224'

def get_model(name, n_output, dataset=None, checkpoint=None, pretrained=True, method_name='', pretrained_cfg=None):
    
    if name == 'vit_b16':
        if 'agc' in method_name or 'rollout' == method_name:
            # timm_model = timm.create_model(MODEL, pretrained=pretrained, num_classes=n_output, pretrained_cfg=pretrained_cfg)
            timm_model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg=pretrained_cfg)
            state_dict = timm_model.state_dict()
            model = ViT_Ours.create_model(MODEL, pretrained=pretrained, num_classes=n_output)
            model.load_state_dict(state_dict, strict=True)
        else:
            model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg=pretrained_cfg)

    print('------- [MODEL Default config] -------')
    print(model.default_cfg)
    print()
 
    return model
