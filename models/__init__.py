from .p2pnet import build
from .ffnet.FFNet import build_ffnet

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if args.arch == 'p2pnet':
        return build(args, training)
    elif args.arch == 'ffnet':
        return build_ffnet(args, training)
    else:
        raise NotImplementedError(f'This model————{args.arch} has not been implemented yet.')
