
from models.backones.vgg.vgg import get_vgg16_net
from models.backones.dla.dla_dcn import get_dla_net

_model_factory = {
  'vgg16': get_vgg16_net, # default Resnet with deconv
  'dla': get_dla_net,
}

def build_backbone(args):
    arch = args.backbone
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_backone = _model_factory[arch]
    backbone = get_backone(args)
    return backbone