"""The built kernels keep the PTX instructions in gpu/spawn/ptx_pins.json: each pinned form with its exact
count, and at most the pinned registers, stack and spills. The pins are the instructions of the inline PTX
that these kernels had before it moved to MLIR ops. Run gpu/spawn/index_mma/build.sh and
gpu/spawn/fp8_gemm/build.sh with PTXAS set first; without a build, the tests skip."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ptx_pins', Path(__file__).resolve().with_name('ptx_pins.py'))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

PINNED = json.loads(P.PINS.read_text())


def test_every_kernel_has_pins():
    assert sorted(PINNED) == sorted(P.KERNELS)


@pytest.mark.parametrize('kernel', P.KERNELS)
def test_kernel_keeps_its_pinned_instructions(kernel):
    ptx_file = P.BUILD / f'{kernel}.ptx'
    if not ptx_file.exists():
        pytest.skip(f'{ptx_file} is not built')
    counts = P.Counter(P.instructions(ptx_file.read_text()))
    got = {form: counts[form] for form in PINNED[kernel]['instructions']}
    assert got == PINNED[kernel]['instructions']


@pytest.mark.parametrize('kernel', P.KERNELS)
def test_kernel_needs_no_more_registers_stack_or_spills(kernel):
    manifest = P.BUILD / f'{kernel}.json'
    if not manifest.exists():
        pytest.skip(f'{manifest} is not built')
    numbers = P.ptxas_numbers(json.loads(manifest.read_text()))
    if numbers is None:
        pytest.skip(f'{kernel} was built without PTXAS')
    registers, stack = numbers
    assert registers <= PINNED[kernel]['max_registers']
    assert stack <= PINNED[kernel]['max_stack_and_spill_bytes']
