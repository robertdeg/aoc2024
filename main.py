import functools
import itertools
import time
from math import sqrt, floor, ceil, lcm
import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    combinations, permutations, cycle
from functools import reduce, partial, cmp_to_key
from collections import Counter, defaultdict
import re
from typing import Iterable, Any

import numpy as np

def day8(filename: str):
    world = {(r, c) : ch for r, line in enumerate(open(filename)) for c, ch in enumerate(line.strip())}
    ants = defaultdict(set)
    for pos, ch in world.items():
        if ch != '.':
            ants[ch].add(pos)

    def expand(r1, c1, r2, c2, s=0):
        return takewhile(world.__contains__, ((r2 + k * (r2 - r1), c2 + k * (c2 - c1)) for k in count(s)))

    locs1 = {next(expand(*p1, *p2, 1), None) for _, ps in ants.items() for p1, p2 in permutations(ps, 2)}
    locs2 = reduce(set.union, (set(expand(*p1, *p2)) for _, ps in ants.items() for p1, p2 in permutations(ps, 2)))

    part1 = len(locs1 - {None})
    part2 = len(locs2)

    return part1, part2

def day7(filename: str):
    def strip_last(a: int, b: int) -> int:
        return int(str(a)[:-len(str(b))])

    def check(total: int, numbers: [int], advanced: bool) -> bool:
        if total < 0:
            return False
        elif not numbers:
            return total == 0
        if total % numbers[-1] == 0:
            if check(total // numbers[-1], numbers[:-1], advanced):
                return True
        if check(total - numbers[-1], numbers[:-1], advanced):
            return True
        if advanced and str(total).endswith(str(numbers[-1])):
            return check(strip_last(total, numbers[-1]), numbers[:-1], advanced)
        else:
            return False


    xss = [list(map(int, re.findall(r"\d+", line))) for line in open(filename).readlines()]
    part1 = sum(tot for tot, *nrs in xss if check(tot, nrs, False))
    part2 = sum(tot for tot, *nrs in xss if check(tot, nrs, True))

    return part1, part2

def day6(filename: str):
    world = {(r, c) : ch for r, line in enumerate(open(filename)) for c, ch in enumerate(line.strip())}
    guard = next(pos for pos in world if world[pos] == '^')

    def forward(r, c, dr, dc) -> (int, int):
        return r + dr, c + dc

    def move(r, c, dr, dc) -> (int, int, int, int):
        r1, c1 = forward(r, c, dr, dc)
        return (r, c, dc, -dr) if world.get((r1, c1), '') == '#' else (r1, c1, dr, dc)

    def trail(r: int, c: int, dr, dc):
        path = set()
        while (r, c, dr, dc) not in path:
            yield r, c, dr, dc
            path.add((r, c, dr, dc))
            r, c, dr, dc = move(r, c, dr, dc)

    route = list(takewhile(lambda p: (p[0], p[1]) in world, trail(*guard, -1, 0)))
    visited = {pos : t for pos, t in zip(route, count())}

    part2 = 0
    path = set()
    for (r, c, dr, dc), t in zip(route, count()):
        rblock, cblock, *_ = forward(r, c, dr, dc)
        if (rblock, cblock) not in path and world.get((rblock, cblock), '') == '.':
            world[(rblock, cblock)] = '#'
            for p in trail(r, c, dc, -dr):
                if (p[0], p[1]) not in world:
                    break
                elif visited.get(p, 100000000) < t:
                    part2 += 1
                    break
            else:
                part2 += 1    # got stuck in a loop
            world[(rblock, cblock)] = '.'
        path.add((r, c))

    part1 = len(path)

    return part1, part2

def day5(filename: str):
    p1, p2 = map(str.splitlines, open(filename).read().split("\n\n"))
    paths = [eval(f"[{line}]") for line in p2]
    out = defaultdict(set)
    for a, b in (tuple(map(int, line.split("|"))) for line in p1):
        out[a].add(b)

    def correct(seq: list[int]) -> bool:
        return not any(a in out[b] for a, b in combinations(seq, 2))

    def dfs(vertices: set, source: int, visited: set = set()) -> Iterable[int]:
        visited.add(source)
        for v in (out[source] & vertices):
            yield from dfs(vertices, v, visited) if v not in visited else ()
        yield source

    def postorder(vertices: set) -> Iterable[int]:
        visited = set()
        yield from chain.from_iterable(dfs(vertices, v, visited) for v in vertices if v not in visited)

    part1 = sum(ys[len(ys) // 2] for ys in paths if correct(ys))
    part2 = sum(next(islice(postorder(set(ys)), len(ys) // 2, None)) for ys in paths if not correct(ys))

    return part1, part2

def day4(filename: str):
    def get(haystack: dict, pos, delta, n) -> str:
        return "".join(haystack.get((pos[0] + i * delta[0], pos[1] + i * delta[1]), '') for i in range(n))

    chars = {(r, c) : ch for r, line in enumerate(open(filename).readlines()) for c, ch in enumerate(line)}
    dirs = [(-1, -1), (-1, 1), (1, 1), (1, -1), (-1, 0), (0, 1), (1, 0), (0, -1)]

    part1 = sum(get(chars, pos, delta, 4) == "XMAS" for pos in chars for delta in dirs)
    part2 = sum({get(chars, (p[0] - d[0], p[1] - d[1]), d, 3) for d in ((1, 1), (1, -1))} <= {"SAM", "MAS"}
                for p in chars)

    return part1, part2

def day3(filename: str):
    muls = (map(int, m.groups()) for m in re.finditer(r"mul\((\d+),(\d+)\)", open(filename).read()))
    instrs = [m.groups() for m in re.finditer(r"mul\((\d+),(\d+)\)|do(n't)?\(\)", open(filename).read())]
    def folder(state, tup):
        s, accepting = state
        a, b, dont = tup
        return (s, dont is None) if a is None else (s + int(a) * int(b), True) if accepting else state


    part1 = sum(a * b for a, b in muls)
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
