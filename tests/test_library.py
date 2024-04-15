import torch

import neuzips


def test_fp32():
    m = neuzips.Manager()
    rt = torch.randn(1024, 1024, 16).cuda()
    result = torch.empty_like(rt)
    e, f = m.split_and_compress(rt)

    compressed_size = e.nbytes + f.nbytes
    original_size = rt.nbytes

    assert compressed_size < original_size

    m.decompress_and_merge(e, f, result)

    assert torch.allclose(rt, result, atol=1e-5)


def test_fp16():
    m = neuzips.Manager()
    rt = torch.randn(1024, 1024, 16).cuda().half()
    result = torch.empty_like(rt)
    e, f = m.split_and_compress(rt)

    compressed_size = e.nbytes + f.nbytes
    original_size = rt.nbytes

    assert compressed_size < original_size

    m.decompress_and_merge(e, f, result)

    assert torch.allclose(rt, result, atol=1e-1)


def test_memory_leak():
    m = neuzips.Manager()

    peak = None
    root = torch.randn(1024, 1024, 16).cuda().to(torch.bfloat16)
    for _ in range(1000):
        rt = torch.randn_like(root)
        result = torch.empty_like(rt)
        e, f = m.split_and_compress(rt)
        m.decompress_and_merge(e, f, result)

        assert e.nbytes + f.nbytes < rt.nbytes

        if peak is None:
            peak = torch.cuda.memory_allocated()
        else:
            assert torch.cuda.memory_allocated() <= (peak * 1.05)
