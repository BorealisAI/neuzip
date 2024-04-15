import logging

import torch

import neuzips

FRACTION_BITS = 7
SHAPE = (1024, 1024, 16)
DEVICE = torch.device("cuda")
STD = 32


def test_lossless(caplog):
    caplog.set_level(logging.INFO)
    m = neuzips.Manager(precision=FRACTION_BITS + 2**8)
    random_tensor = torch.randn(size=SHAPE, dtype=torch.bfloat16, device=DEVICE) / STD
    result = torch.empty_like(random_tensor)
    exponents, fractions = m.split_and_compress(random_tensor)
    compressed_size = exponents.nbytes + fractions.nbytes
    original_size = random_tensor.nbytes

    assert compressed_size < original_size
    logging.info(f"Original size: {original_size}")
    logging.info(f"Compressed size: {compressed_size}")
    logging.info(f"Compression ratio: {compressed_size / original_size}")

    m.decompress_and_merge(exponents, fractions, result)

    logging.info("Target absolute tolerance: 0.0")
    logging.info(f"Empirical absolute tolerance: {torch.abs(result - random_tensor).max().item()}")

    assert torch.equal(random_tensor, result)


def test_lossy(caplog):
    caplog.set_level(logging.INFO)
    random_tensor = torch.randn(size=SHAPE, dtype=torch.bfloat16, device=DEVICE) / STD
    algo = neuzips.Algorithm.ans

    for prec in [30, 20, 10, 0]:
        logging.info(f"Testing with {prec} bits of precision")
        result = torch.empty_like(random_tensor)
        m = neuzips.Manager(algorithm=algo, precision=prec)
        exponents, fractions = m.split_and_compress(random_tensor)
        compressed_size = exponents.nbytes + fractions.nbytes
        original_size = random_tensor.nbytes

        logging.info(f"Original size: {original_size}")
        logging.info(f"Compressed exp size: {exponents.nbytes}")
        logging.info(f"Compressed frac size: {fractions.nbytes}")
        logging.info(f"Compression ratio: {compressed_size / original_size}")

        assert compressed_size < original_size

        m.decompress_and_merge(exponents, fractions, result)

        logging.info(f"Target absolute tolerance: {2**-(prec)}")
        logging.info(f"Empirical absolute tolerance: {torch.abs(result - random_tensor).max().item()}")
        assert torch.allclose(result, random_tensor, atol=2**-(prec), rtol=0)
