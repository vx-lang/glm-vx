"""The PTX instructions that a kernel must keep, so that moving its inline PTX to MLIR ops changes nothing.

`pins(ptx)` gives each special instruction form (an `mma`, `shfl`, `cvt`, load, store and so on, with all its
qualifiers) that the kernel's inline-asm regions use, and the count of that form in the whole kernel.
test_ptx_pins.py checks the built PTX against gpu/spawn/ptx_pins.json. To record new pins after a change
that is meant to change the PTX, run: python3 gpu/spawn/ptx_pins.py --update
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = Path(__file__).with_name('ptx_pins.json')
BUILD = ROOT / 'gpu' / 'build' / 'spawn'
# The kernels built from the sources whose PTX moved to MLIR ops, relative to BUILD.
KERNELS = [f'index_mma/{n}' for n in (
    'index_key_bf16_at', 'index_key_fp8_at', 'index_q_bf16_at', 'index_scores_chunk_at',
    'index_scores_chunk_hist_at', 'index_scores_fp8_at', 'index_scores_fp8_hist_at', 'index_scores_mma16_at',
    'index_scores_mma16_hist_at', 'index_scores_mma_at', 'index_scores_mma_hist_at', 'topk_mb_list_at',
    'topk_mb_write_at', 'topk_select_ilp_at', 'topk_select_ilp_hist_at', 'topk_select_warp_at')] + [
    f'fp8_gemm/fp8_gemm_{n}' for n in ('grouped_m64', 'm64', 'grouped_plan', 'reduce')]
# Plain arithmetic, moves and compares are left out: LLVM may choose them differently and be right.
SPECIAL = re.compile(r'^(mma|wgmma|shfl|vote|match|cvt|cp\.async|mbarrier|fence|prefetch|bar|popc|div\.rn|ld|st)\.')


def instructions(text):
    """The instruction forms in PTX text: the opcode with its qualifiers, without a predicate."""
    for part in re.split(r'[;\n]', text.replace('{', ' ').replace('}', ' ')):
        part = re.sub(r'^\s*@!?%?\w+\s+', '', part).strip()
        if part and not part.startswith(('.', '//', '$')) and not part.endswith(':'):
            yield part.split()[0]


def pins(ptx):
    """Each special form that the inline-asm regions of `ptx` use, and its count in all of `ptx`."""
    asm = ''.join(re.findall(r'// begin inline asm\n(.*?)// end inline asm', ptx, re.S))
    forms = {f for f in instructions(asm) if SPECIAL.match(f)}
    counts = Counter(instructions(ptx))
    return {f: counts[f] for f in sorted(forms)}


def ptxas_numbers(manifest):
    """The registers and the stack bytes in a build's ptxas log, or None when ptxas did not run."""
    log = ' '.join(manifest.get('ptxas_log') or [])
    regs = re.search(r'Used (\d+) registers', log)
    stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    if not regs or not stack:
        return None
    return int(regs[1]), sum(int(n) for n in stack.groups())


def record(build):
    out = {}
    for kernel in KERNELS:
        ptx = (build / f'{kernel}.ptx').read_text()
        numbers = ptxas_numbers(json.loads((build / f'{kernel}.json').read_text()))
        if numbers is None:
            sys.exit(f'{kernel}: build with PTXAS set, so that the pins have the register counts')
        out[kernel] = {'instructions': pins(ptx), 'max_registers': numbers[0], 'max_stack_and_spill_bytes': numbers[1]}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--update', action='store_true', help='write ptx_pins.json from the built PTX')
    parser.add_argument('--build', type=Path, default=BUILD)
    args = parser.parse_args()
    if args.update:
        PINS.write_text(json.dumps(record(args.build), indent=1, sort_keys=True) + '\n')
