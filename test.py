import subprocess

# 定义要执行的命令
commands = [
#     # Neurotoxin 攻击者数量为2 聚合方法分别为avg clip robustLR CRFL 数据集为cifar100
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '2', '--agg_method', 'avg', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '2', '--agg_method', 'clip', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '2', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '2', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     # Neurotoxin 攻击者数量为1 聚合方法分别为avg clip robustLR CRFL 数据集为cifar100
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '1', '--agg_method', 'avg', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '1', '--agg_method', 'clip', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '1', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'Neurotoxin',
#      '--num_adversaries', '1', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     # DBA 攻击者数量为4 聚合方法分别为avg clip robustLR CRFL 数据集为cifar100
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'DBA',
#      '--num_adversaries', '4', '--agg_method', 'avg', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'DBA',
#      '--num_adversaries', '4', '--agg_method', 'clip', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'DBA',
#      '--num_adversaries', '4', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'DBA',
#      '--num_adversaries', '4', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     # CerP 攻击者数量为2 聚合方法分别为avg clip robustLR CRFL 数据集为cifar100
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '2', '--agg_method', 'avg', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '2', '--agg_method', 'clip', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '2', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '2', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     # CerP 攻击者数量为1 聚合方法分别为avg clip robustLR CRFL 数据集为cifar100
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '1', '--agg_method', 'avg', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '1', '--agg_method', 'clip', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '1', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'CerP',
#      '--num_adversaries', '1', '--agg_method', 'CRFL', '--dataset', 'cifar100'],

#     # My A3FL 攻击者数量为4/2/1 聚合方法CRFL 数据集为cifar100/10
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
#      '--num_adversaries', '4', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'A3FL',
#      '--num_adversaries', '4', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
#      '--num_adversaries', '2', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'A3FL',
#      '--num_adversaries', '2', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
#      '--num_adversaries', '1', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
#     ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'A3FL',
#      '--num_adversaries', '1', '--agg_method', 'CRFL', '--dataset', 'cifar100'],
    
    
    ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
     '--num_adversaries', '1', '--agg_method', 'avg', '--dataset', 'cifar100'],
    ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
     '--num_adversaries', '1', '--agg_method', 'clip', '--dataset', 'cifar100'],
    ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
     '--num_adversaries', '1', '--agg_method', 'robustLR', '--dataset', 'cifar100'],
    ['python', 'clean.py', '--gpu', '0', '--params', 'configs/config.yaml', '--attack_method', 'My',
     '--num_adversaries', '1', '--agg_method', 'CRFL', '--dataset', 'cifar100']
]

# 依次执行每个命令
for command in commands:
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"Executed command: {' '.join(command)}")
    if result.returncode == 0:
        print("Command executed successfully.")
        # # 如果需要查看标准输出
        # print("Output:\n", result.stdout)
    else:
        print("Error executing command.")
        # 如果需要查看错误信息
        print("Error:\n", result.stderr)
