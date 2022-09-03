import os
import pprint

from trainer import Trainer
from utils import get_command_line_parser, postprocess_args


_utils_pp = pprint.PrettyPrinter()
def _pprint(x):
    _utils_pp.pprint(x)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
    

if __name__ == '__main__':
    
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    _pprint(vars(args))

    # 检查挂载的数据路径是否存在
    if os.path.exists("/apdcephfs/share_1324356/data/videoqa/scenic_spot/TLDv3/"):
        with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
            f.write("path exist!\n" )
    else:
        with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
            f.write("path not exist!\n" )

    # 提交平台运行不需要手动指定gpu号了
    # set_gpu(args.gpu)
    # with open(os.path.join(args.save_path, 'record.txt'), 'a') as f:
    #     f.write("using gpu: {}\n".format(args.gpu) )
    

    trainer = Trainer(args)
    trainer.train()
    trainer.test()
    print(args.save_path)
