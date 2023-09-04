import argparse
class M(object):
    def __init__(self, *args, **kwargs):
        print(kwargs)
        print(kwargs['config'])
        print(kwargs.config)
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/frozen/config/train.yaml')
    parser.add_argument('--config_name', type=str, default='base')
    parser.add_argument('--percent', type=float, default=-1)
    parser.add_argument('--only_vision', action='store_true', help='using only vision model')
    parser.add_argument('--without_rationale', action='store_true', help='using only without rationale')
    parser.add_argument('--answerlossonly', action='store_true', help='using onlyanswerloss only')
    parser.add_argument('--pretrained_path', type=str, default=None)
    args = parser.parse_args()
    
    return args
args = get_args()
print(args)
make_object = M(3, **vars(args))