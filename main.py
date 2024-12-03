import functools
import itertools
import time
from math import sqrt, floor, ceil, lcm
import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    combinations, permutations, cycle
from functools import reduce, partial, cmp_to_key
from collections import Counter
import re
from typing import Iterable, Any

import numpy as np


def day3(filename: str):
    muls = (int(m.group(1)) * int(m.group(2)) for m in re.finditer(r"mul\((\d+),(\d+)\)", open(filename).read()))
    instrs = [m.groups() for m in re.finditer(r"(mul\((\d+),(\d+)\))|do(n't)?\(\)", open(filename).read())]
    def folder(state, tup):
        s, accepting = state
        mul, a, b, dont = tup
        return (s, dont is None) if mul is None else (s + int(a) * int(b), True) if accepting else state


    part1 = sum(muls)
    part2, _ = reduce(folder, instrs, (0, True))

    return part1, part2

def day2(filename: str):
    def safe(nrs: list[int]) -> bool:
        return all(0 < a - b < 4 for a, b in zip(nrs[1:], nrs)) or all(0 < b - a < 4 for a, b in zip(nrs[1:], nrs))

    numbers = [list(map(int, line.split(" "))) for line in open(filename).readlines()]

    part1 = sum(map(safe, numbers))
    part2 = sum(any(safe(nrs[:i] + nrs[i+1:]) for i in range(len(nrs))) for nrs in numbers)

    return part1, part2

def day1(filename: str):
    numbers = [map(int, line.split("  ")) for line in open(filename).readlines()]
    left, right = zip(*numbers)
    ls, rs = sorted(left), sorted(right)

    freqs = Counter(rs)

    part1 = sum(abs(l - r) for l, r in zip(ls, rs))
    part2 = sum(l * freqs[l] for l in ls)

    return part1, part2


if __name__ == '__main__':
    solvers = [(key, value) for key, value in globals().items() if key.startswith("day") and callable(value)]
    solvers = sorted(((int(key.split('day')[-1]), value) for key, value in solvers), reverse=True)

    for idx, solver in solvers:
        ns1 = time.process_time_ns()
        p1, p2 = solver(f"input/day{idx}.txt")
        ns2 = time.process_time_ns()
        print(f"day {idx} - part 1: {p1}, part 2: {p2}. time: {(ns2 - ns1) * 1e-9} seconds")
        # break
