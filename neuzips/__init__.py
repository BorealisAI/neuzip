import enum
import logging
from typing import Optional

import torch

try:
    import neuzips_cuda
except ImportError:
    raise ImportError("The CUDA extension is not built. Please run `python setup.py install`")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _convert(manager, module, prefix=""):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            logger.debug(f"Converting {full_name}: {child}")
            setattr(module, name, Linear(manager, child))
        else:
            _convert(manager, child, f"{prefix}.{name}" if prefix else name)
    return module


class CompressedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, _p):
        weight = _p()
        output = torch.nn.functional.linear(inputs, weight)
        ctx.save_for_backward(inputs)
        ctx.carry = (_p,)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (inputs,) = ctx.saved_tensors
        (_p,) = ctx.carry

        weight = _p()

        grad_inputs = torch.nn.functional.linear(grad_output, weight.t())
        grad_weight = torch.matmul(
            grad_output.reshape(-1, grad_output.shape[-1]).t(), inputs.reshape(-1, inputs.shape[-1])
        )

        if _p._gradient is not None:
            _p._gradient += grad_weight
        else:
            _p._gradient = grad_weight
        return grad_inputs, None


class Algorithm(enum.Enum):
    ans = neuzips_cuda.Algorithm.ans
    gdeflate = neuzips_cuda.Algorithm.gdeflate
    zstd = neuzips_cuda.Algorithm.zstd


class Parameter(torch.nn.Parameter):
    def __new__(cls, manager, data=None, requires_grad=True) -> "Parameter":
        p = super().__new__(cls, torch.zeros(0), requires_grad)
        if data is None:
            data = torch.empty(0)

        p._handle = manager.write(data.cuda())
        p._gradient = None
        p._manager = manager
        p._shape = data.shape

        del data
        return p

    def __call__(self) -> torch.Tensor:
        return self._manager.read(self._handle).reshape(self._shape).to(self.device)


class Linear(torch.nn.Module):
    def __init__(self, manager, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.manager = manager
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

        self._p = Parameter(self.manager, linear.weight)

    def extra_repr(self) -> str:
        return f"! in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x):
        if self.bias is None:
            return CompressedLinear.apply(x, self._p)
        else:
            return CompressedLinear.apply(x, self._p) + self.bias


class Manager:
    def __init__(self, algorithm: Algorithm = Algorithm.ans, precision: int = 3) -> None:
        if precision == 7:
            self._be = neuzips_cuda.ManagerM7(algorithm.value)
        elif precision == 3:
            self._be = neuzips_cuda.ManagerM3(algorithm.value)
        elif precision == 1:
            self._be = neuzips_cuda.ManagerM1(algorithm.value)
        else:
            raise ValueError("Precision must be 1, 3, or 7")

        self.meta_dict = {}

    def write(self, tensor: torch.Tensor, handle: Optional[str] = None) -> str:
        if handle is None:
            handle = str(len(self.meta_dict))
            self.meta_dict[handle] = tensor.shape

        self._be.write(handle, tensor)

        return handle

    def read(self, handle: str) -> torch.Tensor:
        shape = self.meta_dict[handle]
        return self._be.read(handle).reshape(shape)

    def convert(self, module: torch.nn.Module) -> torch.nn.Module:
        return _convert(self, module)


if __name__ == "__main__":
    m = Manager()
    rt = torch.randn((1024, 1024))

    handle = m.write(rt)
    rrt = m.read(handle)

    assert torch.allclose(rt, rrt.cpu(), atol=2**-20)

    linear = torch.nn.Linear(1024, 1024)

    nlinear = Linear(m, linear)
    target_result = linear(rt)
    test_result = nlinear(rt)

    assert torch.allclose(test_result, target_result, atol=2**-10, rtol=2**-5)

    target_loss = target_result.mean()
    target_loss.backward()

    test_loss = test_result.mean()
    test_loss.backward()

    print(test_loss, target_loss)

    assert torch.allclose(linear.weight.grad, nlinear._p._gradient, atol=2**-10, rtol=2**-5)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    nested = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
    )
    target_result = nested(rt)
    target_loss = target_result.mean()
    target_loss.backward()

    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**20} MB")

    nested = m.convert(nested).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    test_result = nested(rt.cuda())

    assert torch.allclose(test_result.cpu(), target_result, atol=2**-10, rtol=2**-5), (test_result, target_result)

    test_loss = test_result.mean()
    test_loss.backward()

    assert torch.allclose(target_loss, test_loss.cpu(), atol=2**-10, rtol=2**-5)

    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 2**20} MB")

    print("All tests passed.")
