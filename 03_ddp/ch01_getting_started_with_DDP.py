import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")
    
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
    
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")


# if __name__ == "__main__":
#     n_gpus = torch.cuda.device_count()
#     assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
#     world_size = n_gpus
#     run_demo(demo_basic, world_size)
#     run_demo(demo_checkpoint, world_size)
#     world_size = n_gpus//2
#     run_demo(demo_model_parallel, world_size)
    
    ##########
    
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic():
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#     dist.init_process_group("nccl")
#     rank = dist.get_rank()
#     print(f"Start running basic DDP example on rank {rank}.")
#     # create model and move it to GPU with id rank
#     device_id = rank % torch.cuda.device_count()
#     model = ToyModel().to(device_id)
#     ddp_model = DDP(model, device_ids=[device_id])
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(device_id)
#     loss_fn(outputs, labels).backward()
#     optimizer.step()
#     dist.destroy_process_group()
#     print(f"Finished running basic DDP example on rank {rank}.")

# if __name__ == "__main__":
#     demo_basic()
    
    
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    """
    DDP 환경 초기화
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """
    DDP 환경 정리
    """
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    """
    DDP 기본 예제 실행 함수
    """
    setup(rank, world_size)
    print(f"Start running basic DDP example on rank {rank}.")

    # 모델 생성 및 DDP 설정
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 손실 함수 및 옵티마이저 설정
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 학습 과정
    optimizer.zero_grad()
    inputs = torch.randn(20, 10).to(rank)
    labels = torch.randn(20, 5).to(rank)
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Finished running basic DDP example on rank {rank}.")
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 사용 가능한 GPU 수
    torch.multiprocessing.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)
