# WikiText-103 Raw 训练配置 - 集群版 (A100)
out_dir = 'out-wikitext103-raw'
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = True
wandb_log = False
wandb_project = 'wikitext103-raw'
wandb_run_name = 'sigmagpt-distilgpt2'

dataset = 'wikitext103_raw'

# DistilGPT2 模型配置 (82M 参数)
n_layer = 6
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.1

# 训练参数
batch_size = 8
gradient_accumulation_steps = 16  # effective batch = 128
learning_rate = 5e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

# ~66M tokens, 64527 chunks, 5 epochs
max_iters = 2500
lr_decay_iters = 2500
min_lr = 5e-5
warmup_iters = 200

# Early stopping: val loss 连续5次eval没明显下降就停止
early_stopping_patience = 5
early_stopping_min_delta = 0.01  # 最小改善阈值

compile = True  # A100 支持 torch.compile
