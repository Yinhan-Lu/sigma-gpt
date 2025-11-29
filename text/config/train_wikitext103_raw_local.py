# WikiText-103 Raw 训练配置 - 本地测试版 (MPS)
out_dir = 'out-wikitext103-raw-local'
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'wikitext103_raw'

# 缩小版模型配置 (~10M 参数，类似 Shakespeare)
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
dropout = 0.1

# 本地训练参数
batch_size = 4
gradient_accumulation_steps = 4  # effective batch = 16
learning_rate = 1e-3
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

# 快速测试：100 iterations
max_iters = 100
lr_decay_iters = 100
min_lr = 1e-4
warmup_iters = 10

compile = False  # MPS 不支持
