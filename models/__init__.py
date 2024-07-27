from .p2pnet import build
from .ffnet.FFNet import build_ffnet
from .ffnet.FFNet2 import build_ffnet2
from .ffnet.FFNet2_1 import build_ffnet2_1
from .ffnet.FFNet2_2 import build_ffnet2_2
from .ffnet.FFNet2_3 import build_ffnet2_3
from .ffnet.FFNet2_4 import build_ffnet2_4
from .ffnet.FFNet3 import build_ffnet3
from .ffnet.FFNet4 import build_ffnet4
from .ffnet.FFNet4_1 import build_ffnet4_1
from .ffnet.FFNet4_2 import build_ffnet4_2

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if args.arch == 'p2pnet':
        return build(args, training)
    elif args.arch == 'ffnet':
        return build_ffnet(args, training)
    elif args.arch == 'ffnet2':
        return build_ffnet2(args, training)
    elif args.arch == 'ffnet2_1':
        return build_ffnet2_1(args, training)
    elif args.arch == 'ffnet2_2':
        return build_ffnet2_2(args, training)
    elif args.arch == 'ffnet2_3':
        return build_ffnet2_3(args, training)
    elif args.arch == 'ffnet2_4':
        return build_ffnet2_4(args, training)
    elif args.arch == 'ffnet3':
        return build_ffnet3(args, training)
    elif args.arch == 'ffnet4':
        return build_ffnet4(args, training)
    elif args.arch == 'ffnet4_1':
        return build_ffnet4_1(args, training)
    elif args.arch == 'ffnet4_2':
        return build_ffnet4_2(args, training)
    else:
        raise NotImplementedError(f'This model————{args.arch} has not been implemented yet.')
