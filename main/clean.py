# import sys

# sys.path.append("../")

# import wandb
# import argparse
# import yaml

# import torch
# import numpy as np
# import random

# from fl_utils.helper import Helper
# from fl_utils.fler import FLer

# import os


# def setup_wandb(config_path, sweep):
#     with open(config_path, 'r') as stream:
#         sweep_configuration = yaml.safe_load(stream)

#     if sweep:
#         sweep_id = wandb.sweep(sweep=sweep_configuration, project='FanL-clean')
#         return sweep_id
#     else:
#         # 获取 YAML 文件中的参数
#         config = sweep_configuration['parameters']
#         d = dict()
#         for k in config.keys():
#             v = config[k][list(config[k].keys())[0]]
#             if type(v) is list:
#                 d[k] = {'value': v[0]}
#             else:
#                 d[k] = {'value': v}

#         # 添加命令行参数到配置
#         # d['dataset'] = {'value': args.dataset}
#         # d['attack_method'] = {'value': args.attack_method}
#         # d['num_adversaries'] = {'value': int(args.num_adversaries)}
#         # d['agg_method'] = {'value': args.agg_method}

#         # 写入到临时 YAML 文件
#         yaml.dump(d, open('./yamls/tmp.yaml', 'w'))

#         # 使用修改后的配置初始化 WandB
#         wandb.init(config='./yamls/tmp.yaml')
#         return None


# def set_seed(seed):
#     # seed = int(time.time())
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def main():
#     run = wandb.init()
#     set_seed(wandb.config.seed)
#     helper = Helper(wandb.config)
#     fler = FLer(helper)
#     fler.train()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--params', default='./yamls/poison.yaml')
#     parser.add_argument('--gpu', default=7)
#     parser.add_argument('--sweep', action='store_true')
#     # parser.add_argument('--attack_method')
#     # parser.add_argument('--num_adversaries')
#     # parser.add_argument('--agg_method')
#     # parser.add_argument('--dataset')
#     args = parser.parse_args()

#     torch.cuda.set_device(int(args.gpu))
#     # 初始化 WandB
#     sweep_id = setup_wandb(args.params, args.sweep)
#     if args.sweep:
#         wandb.agent(sweep_id, function=main, count=1)
#     else:
#         main()
import sys

sys.path.append("../")

import argparse
import yaml

import torch
import numpy as np
import random

from fl_utils.helper import Helper
from fl_utils.fler import FLer

import os


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
    
    # 提取参数配置
    if 'parameters' in config_data:
        config = config_data['parameters']
        # 将wandb格式的配置转换为普通字典
        parsed_config = {}
        for k, v in config.items():
            if isinstance(v, dict):
                # 获取第一个键的值
                first_key = list(v.keys())[0]
                value = v[first_key]
                if isinstance(value, list):
                    parsed_config[k] = value[0]
                else:
                    parsed_config[k] = value
            else:
                parsed_config[k] = v
        return parsed_config
    else:
        # 如果配置文件格式不是wandb格式，直接返回
        return config_data


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SimpleConfig:
    """简单的配置类，用于替代wandb.config"""
    def __init__(self, config_dict):
        self._config = config_dict
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        """实现字典式的get方法"""
        return self._config.get(key, default)
    
    def __getitem__(self, key):
        """支持字典式访问 config['key']"""
        return self._config[key]
    
    def __setitem__(self, key, value):
        """支持字典式设置 config['key'] = value"""
        self._config[key] = value
        setattr(self, key, value)
    
    def __contains__(self, key):
        """支持 'key' in config"""
        return key in self._config
    
    def keys(self):
        """返回所有键"""
        return self._config.keys()
    
    def values(self):
        """返回所有值"""
        return self._config.values()
    
    def items(self):
        """返回所有键值对"""
        return self._config.items()
    
    def to_dict(self):
        """转换为字典"""
        return self._config.copy()


def main(config_path):
    # 加载配置
    config_dict = load_config(config_path)
    config = SimpleConfig(config_dict)
    
    # 设置随机种子
    seed = config.get('seed', 42)  # 现在支持get方法了
    set_seed(seed)
    
    # 初始化helper和fler
    helper = Helper(config)
    fler = FLer(helper)
    fler.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='./yamls/poison.yaml')
    parser.add_argument('--gpu', default=7)
    args = parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    
    # 直接运行主函数
    main(args.params)

