"""
src/mpu.py
"""
import os
import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import Function


class MPU(object):
    """
    MPU: Model Parallel Unit

    Notes:
        Let's say we have a total of 16 GPUs denoted g0 ... g15 and we use 2 GPUs to parallelize the model tensor,
        and 4 GPUs to parallelize the model pipeline. The present method will create 8 tensor model-parallel group,
        4 pipeline model parallel groups and 8 data parallel groups as:

        - width: 4 pipeline parallel group
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        - height: 8 tensor parallel group
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        - depth: 8 data parallel group
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]

                        [g02, g06, g10, g14]
                      /  |              /  |
                     [g00, g04, g08, g12]  |
                     |   |             |   |
        3D parallel  |  [g03, g07, g11, g15]
                     |  /              |  /
                     [g01, g05, g09, g13]

                      +---------+  +---------+  +---------+  +---------+
              tensor  |   g00   |  |   g04   |  |   g08   |  |   g14   |
        data          +---------+  +---------+  +---------+  +---------+ ===> forward
              tensor  |   g01   |  |   g05   |  |   g09   |  |   g13   |
                      +---------+  +---------+  +---------+  +---------+
                        pipeline     pipeline     pipeline     pipeline

                      +---------+  +---------+  +---------+  +---------+
              tensor  |   g02   |  |   g06   |  |   g10   |  |   g12   |
        data          +---------+  +---------+  +---------+  +---------+ ===> forward
              tensor  |   g03   |  |   g07   |  |   g11   |  |   g15   |
                      +---------+  +---------+  +---------+  +---------+
                        pipeline     pipeline     pipeline     pipeline

    References:
        Original MPU implementation of Megatron-LM.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/initialize.py

    """

    _tensor_model_parallel_group = None
    _pipeline_model_parallel_group = None
    _data_parallel_group = None

    _tensor_model_parallel_world_size = None
    _pipeline_model_parallel_world_size = None
    _data_parallel_world_size = None

    _tensor_model_parallel_rank = None
    _pipeline_model_parallel_rank = None
    _pipeline_global_ranks = None

    def __init__(
        self,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        backend: str,
        master_port: int,
    ) -> None:
        """
        Initialize MPU object. All process groups are initialized in this method.

        Args:
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
        """

        if not dist.is_initialized():
            self.initialize_distributed(backend, master_port)

        current_rank = dist.get_rank()
        global_world_size = dist.get_world_size()

        assert (
            global_world_size >= tensor_model_parallel_size
        ), "param `tensor_model_parallel_size` must be smaller than global world size."

        assert (
            global_world_size >= pipeline_model_parallel_size
        ), "param `pipeline_model_parallel_size` must be smaller than global world size."

        total_model_parallel_size = (
            tensor_model_parallel_size * pipeline_model_parallel_size
        )

        assert (
            global_world_size % total_model_parallel_size == 0
        ), "global world sizes must be divisible by model parallel world sizes (tp * pp)"

        num_tensor_model_parallel_groups = (
            global_world_size // tensor_model_parallel_size
        )

        num_pipeline_model_parallel_groups = (
            global_world_size // pipeline_model_parallel_size
        )

        # 1. initialize data parallel group
        self._initialize_data_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 2. initialize tensor model parallel group
        self._initialize_tensor_model_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            num_tensor_model_parallel_groups=num_tensor_model_parallel_groups,
        )

        # 3. initialize pipeline model parallel group
        self._initialize_pipeline_model_parallel_group(
            current_rank=current_rank,
            global_world_size=global_world_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 4. create distributed functions
        functions = self._initialize_functions()
        self._broadcast_fn = functions["broadcast"]
        self._reduce_fn = functions["reduce"]
        self._scatter_fn = functions["scatter"]
        self._gather_fn = functions["gather"]

    def _initialize_data_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize data parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
            num_pipeline_model_parallel_groups (int): the number of pipeline model parallel groups
        """
        assert (
            self._data_parallel_group is None
        ), "data parallel group is already initialized."

        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups

            for j in range(tensor_model_parallel_size):
                ranks = list(
                    range(start_rank + j, end_rank, tensor_model_parallel_size)
                )

                group = dist.new_group(ranks)
                if current_rank in ranks:
                    self._data_parallel_group = group

    def _initialize_tensor_model_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        num_tensor_model_parallel_groups: int,
    ) -> None:
        """
        Initialize tensor model parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            num_tensor_model_parallel_groups (int): the number of tensor model parallel groups
        """
        assert (
            self._tensor_model_parallel_group is None
        ), "tensor model parallel group is already initialized."

        for i in range(num_tensor_model_parallel_groups):
            start_rank = i * tensor_model_parallel_size
            end_rank = (i + 1) * tensor_model_parallel_size

            ranks = list(range(start_rank, end_rank))
            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._tensor_model_parallel_group = group

    def _initialize_pipeline_model_parallel_group(
        self,
        current_rank: int,
        global_world_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize pipeline model parallel group

        Args:
            current_rank (int): current rank
            global_world_size (int): global world size
            num_pipeline_model_parallel_groups (int): the number of model parallel groups
        """
        assert (
            self._pipeline_model_parallel_group is None
        ), "pipeline model parallel group is already initialized."

        for i in range(num_pipeline_model_parallel_groups):
            ranks = list(
                range(i, global_world_size, num_pipeline_model_parallel_groups)
            )

            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._pipeline_model_parallel_group = group
                self._pipeline_global_ranks = ranks

    def model_parallel_is_initialized(self) -> bool:
        """
        Check if model and data parallel groups are initialized.

        Returns:
            bool: whether MPU is initialized
        """
        if (
            self._tensor_model_parallel_group is None
            or self._pipeline_model_parallel_group is None
            or self._data_parallel_group is None
        ):
            return False
        return True

    def get_model_parallel_group(self):
        """
        Get the tensor model parallel group.

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_group()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            ProcessGroup: tensor model parallel group
        """
        return self.get_tensor_model_parallel_group()

    def get_model_parallel_world_size(self) -> int:
        """
        Get the tensor model parallel world size

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_world_size()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_world_size()

    def get_model_parallel_rank(self) -> int:
        """
        Get the tensor model parallel rank

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_rank()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_rank()

    def get_tensor_model_parallel_group(self):
        """
        Get tensor model parallel group

        Returns:
            ProcessGroup: tensor model parallel group
        """

        assert (
            self._tensor_model_parallel_group is not None
        ), "tensor model parallel group is not initialized."

        return self._tensor_model_parallel_group

    def get_pipeline_model_parallel_group(self):
        """
        Get pipeline model parallel group

        Returns:
            ProcessGroup: pipeline model parallel group
        """
        assert (
            self._pipeline_model_parallel_group is not None
        ), "pipeline model parallel group is not initialized."

        return self._pipeline_model_parallel_group

    def get_data_parallel_group(self):
        assert (
            self._data_parallel_group is not None
        ), "data parallel group is not initialized."

        return self._data_parallel_group

    def get_tensor_model_parallel_world_size(self) -> int:
        """
        Get tensor model parallel world size

        Returns:
            int: tensor model parallel world size
        """
        if self._tensor_model_parallel_world_size is not None:
            return self._tensor_model_parallel_world_size

        return dist.get_world_size(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set tensor model parallel world size

        Args:
            world_size (int): tensor model parallel world size
        """
        self._tensor_model_parallel_world_size = world_size

    def get_pipeline_model_parallel_world_size(self) -> int:
        """
        Get pipeline model parallel world size

        Returns:
            int: pipeline model parallel world size
        """
        if self._pipeline_model_parallel_world_size is not None:
            return self._pipeline_model_parallel_world_size

        return dist.get_world_size(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set pipeline model parallel world size

        Args:
            world_size (int): pipeline model parallel world size
        """
        self._pipeline_model_parallel_world_size = world_size

    def get_tensor_model_parallel_rank(self) -> int:
        """
        Get tensor model parallel rank

        Returns:
            int: tensor model parallel rank
        """
        if self._tensor_model_parallel_rank is not None:
            return self._tensor_model_parallel_rank

        return dist.get_rank(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_rank(self, rank: int) -> None:
        """
        Set tensor model parallel rank

        Args:
            rank (int): tensor model parallel rank
        """

        self._tensor_model_parallel_rank = rank

    def get_pipeline_model_parallel_rank(self) -> int:
        """
        Get pipeline model parallel rank

        Returns:
            int: pipeline model parallel rank
        """
        if self._pipeline_model_parallel_rank is not None:
            return self._pipeline_model_parallel_rank

        return dist.get_rank(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_rank(self, rank: int) -> None:
        """
        Set pipeline model parallel rank

        Args:
            rank (int): pipeline model parallel rank
        """

        self._pipeline_model_parallel_rank = rank

    def is_pipeline_fist_stage(self) -> bool:
        """
        Return `True` if in the first pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is first
        """
        return self.get_pipeline_model_parallel_rank() == 0

    def is_pipeline_last_stage(self) -> bool:
        """
        Return `True` if in the last pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is last
        """
        return self.get_pipeline_model_parallel_rank() == (
            self.get_pipeline_model_parallel_world_size() - 1
        )

    def get_tensor_model_parallel_src_rank(self) -> int:
        """
        Calculate the global rank corresponding to the first local rank in the tensor model parallel group.

        Returns:
            int: tensor model parallel source rank
        """

        global_rank = dist.get_rank()
        local_world_size = self.get_tensor_model_parallel_world_size()
        return (global_rank // local_world_size) * local_world_size

    def get_pipeline_model_parallel_fist_rank(self):
        """
        Get the first pipeline model parallel rank

        Returns:
            int: the first pipeline model parallel rank
        """
        return self._pipeline_global_ranks[0]

    def get_pipeline_model_parallel_last_rank(self):
        """
        Get the last pipeline model parallel rank

        Returns:
            int: the last pipeline model parallel rank
        """
        return self._pipeline_global_ranks[
            self.get_pipeline_model_parallel_world_size() - 1
            ]

    def get_pipeline_model_parallel_next_rank(self) -> int:
        """
        Get the next pipeline model parallel rank comparison with current stage.

        Returns:
            int: the next pipeline model parallel rank
        """
        assert (
            self._pipeline_global_ranks is not None
        ), "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self) -> int:
        """
        Get the previous pipeline model parallel rank comparison with current stage.

        Returns:
            int: the previous pipeline model parallel rank
        """
        assert (
            self._pipeline_global_ranks is not None
        ), "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe - 1) % world_size]

    def get_data_parallel_world_size(self) -> int:
        """
        Get data parallel world size

        Returns:
            int: data parallel world size
        """

        return dist.get_world_size(self.get_data_parallel_group())

    def get_data_parallel_rank(self) -> int:
        """
        Get data parallel rank

        Returns:
            int: data parallel rank
        """
        return dist.get_rank(self.get_data_parallel_group())

    def destroy_model_parallel(self) -> None:
        """
        Destroy all the model parallel groups
        """

        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_group = None
        self._data_parallel_group = None

    def _broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: broadcast tensor
        """
        return inputs.clone()

    def _reduce(self, inputs: Tensor):
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """
        if self.get_tensor_model_parallel_world_size() == 1:
            return inputs

        dist.all_reduce(inputs, group=self.get_tensor_model_parallel_group())
        return inputs

    def _scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        last_dim_size = inputs.size()[last_dim] // world_size

        inputs_list = torch.split(
            tensor=inputs,
            split_size_or_sections=last_dim_size,
            dim=last_dim,
        )

        rank = self.get_tensor_model_parallel_rank()
        outputs = inputs_list[rank].contiguous()
        return outputs

    def _gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        rank = self.get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(inputs) for _ in range(world_size)]
        tensor_list[rank] = inputs
        torch.distributed.all_gather(
            tensor_list, inputs, group=self.get_tensor_model_parallel_group()
        )
        outputs = torch.cat(tensor_list, dim=last_dim).contiguous()
        return outputs

    def broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor):

        Returns:
            Tensor: broadcast tensor
        """

        if self._enable_grad(inputs):
            outputs = self._broadcast_fn.apply(inputs)
        else:
            outputs = self._broadcast(inputs)
        return outputs

    def reduce(self, inputs: Tensor) -> Tensor:
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """

        if self._enable_grad(inputs):
            outputs = self._reduce_fn.apply(inputs)
        else:
            outputs = self._reduce(inputs)
        return outputs

    def scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._scatter_fn.apply(inputs)
        else:
            outputs = self._scatter(inputs)
        return outputs

    def gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._gather_fn.apply(inputs)
        else:
            outputs = self._gather(inputs)
        return outputs

    @staticmethod
    def _enable_grad(inputs: Tensor) -> bool:
        """
        Check current tensor is enabled to pass gradient.

        Args:
            inputs (Tensor): input tensor

        Returns:
            bool: whether gradient can be passed or not
        """
        return torch.is_grad_enabled() and inputs.requires_grad

    def _initialize_functions(self):
        class Broadcast(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._broadcast(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._reduce(inputs)

        class Reduce(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._reduce(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._broadcast(inputs)

        class Scatter(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._scatter(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._gather(inputs)

        class Gather(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._gather(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._scatter(inputs)

        return {
            "broadcast": Broadcast,
            "reduce": Reduce,
            "scatter": Scatter,
            "gather": Gather,
        }

    @staticmethod
    def initialize_distributed(backend, master_port):
        """Initialize torch.distributed and mpu."""
        if not torch.distributed.is_initialized():
            rank = int(os.getenv("RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 1))
            os.environ["MASTER_PORT"] = str(master_port)
            device_count = torch.cuda.device_count()

            if device_count > 0:
                device = rank % device_count
                torch.cuda.set_device(device)

            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )