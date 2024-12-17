import functools
import heapq
import itertools
import time
from fractions import Fraction
from hashlib import file_digest
from math import sqrt, floor, ceil, lcm
import collections
import operator
from itertools import zip_longest, starmap, count, chain, islice, takewhile, accumulate, tee, dropwhile, repeat, \
    combinations, permutations, cycle, filterfalse, pairwise
from functools import reduce, partial, cmp_to_key, cache
from collections import Counter, defaultdict
import re
from numbers import Rational
from typing import Iterable, Any

import numpy as np


def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = extended_gcd(b % a, a)

    x = y1 - (b // a) * x1
    y = x1

    return gcd, x, y

def solve(a: int, b: int, m: int) -> int:
    # solve a * x = b (mod m)
    g, x1, x2 = extended_gcd(a, m)
    return x1 * b

def day14(filename: str):
    data = [[int(n) for n in re.findall(r"-?\d+", line)] for line in open(filename)]
    bots = [((x, y), (dx, dy)) for x, y, dx, dy in data]

    def draw(bots, width , height, t):
        counts = Counter(pos_at(width, height, x, y, dx, dy, t) for (x, y), (dx, dy) in bots)
        for y in range(height):
            for x in range(width):
                print(f"{counts.get((x, y), '.')}", end="")
            print()

    def pos_at(width, height, x: int, y: int, dx: int, dy: int, t: int):
        return (x + t * dx) % width, (y + t * dy) % height

    def quadrant(width, height, x: int, y: int) -> int:
        qx = 2 * x // width if x != width // 2 else None
        qy = 2 * y // height if y != height // 2 else None
        return qy * 2 + qx if qx is not None and qy is not None else None

    def when_neighbours(pos1, vel1, pos2, vel2):
        th = solve((vel1[0] - vel2[0]) % 101, (pos2[0] - pos1[0] - 1) % 101, 101) % 101 # t = th + k * 101
        k = solve((101 * (vel1[1] - vel2[1])) % 103, (pos2[1] - pos1[1] + th * (vel2[1] - vel1[1])) % 103, 103)
        return (th + k * 101) % (101 * 103)

    scores = Counter(quadrant(101, 103, *pos_at(101, 103, x, y, dx, dy, 100)) for (x, y), (dx, dy) in bots)
    part1 = reduce(operator.mul, (val for key, val in scores.items() if key is not None))

    scores = Counter()
    # for each pair of bots, find out at which time they're each other's horizontal neighbors
    for pos1, v1, pos2, v2 in permutations(bots, 2):
        scores[when_neighbours(pos1, v1, pos2, v2)] += 1

    part2, _ = scores.most_common(1)[0]
    # draw(bots, 101, 103, part2)

    return part1, part2


def day13(filename: str):
    data = [[int(nr) for nr in re.findall(r"\d+", block)] for block in open(filename).read().split("\n\n")]

    def costs(x1, x2, x, y1, y2, y):
        gcd, m, n = extended_gcd(x1, x2)    # find m, n, gcd so that x1 * m + x2 * n == gcd
        mul = x // gcd
        a0, b0 = mul * m, mul * n
        r = Fraction(y - a0 * y1 - b0 * y2, x2 // gcd * y1 - x1 // gcd * y2)
        return 3 * (a0 + r.numerator * x2 // gcd) + b0 - r.numerator * x1 // gcd if r.denominator == 1 else 0

    part1 = sum(costs(dx1, dx2, x, dy1, dy2, y) for dx1, dy1, dx2, dy2, x, y in data)
    part2 = sum(costs(dx1, dx2, x + 10000000000000, dy1, dy2, y + 10000000000000) for dx1, dy1, dx2, dy2, x, y in data)

    return part1, part2

def day12(filename: str):
    world = {(int(r), int(c)) : ch for r, line in enumerate(open(filename)) for c, ch in enumerate(line.strip())}
    def neighbours(r, c):
        yield from ((r - 1, c, 'N'), (r, c + 1, 'E'), (r + 1, c, 'S'), (r, c - 1, 'W'))

    def region(r, c, visited):
        visited.add((r, c))
        area, perimeter = 1, set()
        for nr, nc, nd in neighbours(r, c):
            if world.get((nr, nc), None) != world[r, c]:
                perimeter.add((nr, nc, nd))
            if (nr, nc) not in visited and world.get((nr, nc), None) == world[r, c]:
                a, p = region(nr, nc, visited)
                area += a
                perimeter.update(p)
        return area, perimeter

    def sides(edges: set, direction: str):
        horz = direction not in "NS"
        xs = sorted(filter(lambda e: e[2] == direction, edges), key=lambda e: (e[horz], e[not horz]))
        return 1 + sum(b[horz] != a[horz] or abs(b[not horz] - a[not horz]) != 1 for a, b in pairwise(xs))

    visited = set()
    info = {pos : region(*pos, visited) for pos in world if pos not in visited}

    part1 = sum(info[pos][0] * len(info[pos][1]) for pos in info)
    part2 = sum(info[pos][0] * sides(info[pos][1], d) for pos, d in itertools.product(info, "NSEW"))
    return part1, part2

def day11(filename: str):
    nrs = list(map(int, re.findall(r"\d+", open(filename).read())))

    def spliteven(nr: str) -> (int, int):
        l = len(nr) // 2
        return int(nr[:l]), int(nr[l:])

    @cache
    def howmany(nr: int, blinks: int):
        if blinks == 0:
            return 1
        if nr == 0:
            return howmany(1, blinks - 1)
        if len(str(nr)) % 2 == 0:
            return sum(howmany(a, blinks - 1) for a in spliteven(str(nr)))
        return howmany(nr * 2024, blinks - 1)

    part1 = sum(howmany(nr, 25) for nr in nrs)
    part2 = sum(howmany(nr, 75) for nr in nrs)

    return part1, part2

def day10(filename: str):
    world = {(r, c) : int(ch) for r, line in enumerate(open(filename)) for c, ch in enumerate(line.strip())}

    def nbors(r, c):
        yield from ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c))

    def paths(p: (int, int), seen, all):
        if not all and p in seen:
            return 0
        seen.add(p)
        return 1 if world[p] == 9 else sum(paths(n, seen, all) for n in nbors(*p)
                                           if (all or n not in seen) and world.get(n, 0) == world[p] + 1)

    part1 = sum(paths(pos, set(), False) for pos in world if world[pos] == 0)
    part2 = sum(paths(pos, set(), True) for pos in world if world[pos] == 0)

    return part1, part2

def day9(filename: str):
    line = [int(d) for d in open(filename).read().strip()]
    disk = [((i // 2 if i % 2 == 0 else None), n) for i, n in enumerate(line)]
    poss = list(accumulate((n for _, n in disk), lambda a, b: a + b, initial=0))
    space = [(pos, size) for pos, (id, size) in zip(poss, disk) if id is None]
    files = {pos : (size, id) for pos, (id, size) in zip(poss, disk) if id is not None}

    def compress(space, files):
        result = dict()
        reads = ((pos, *files[pos]) for pos in reversed(files))
        writer = iter(space)
        writepos, free = next(writer)
        read_pos, size, fid = next(reads)
        while writepos < read_pos + size:
            moved = min(free, size)
            result[writepos] = (moved, fid)
            writepos += moved
            free, size = free - moved, size - moved
            if free == 0:
                writepos, free = next(writer)
            if size == 0:
                read_pos, size, fid = next(reads)
        result[read_pos] = size, fid
        result.update({read_pos : (size, fid) for read_pos, size, fid in reads})
        return {(pos, *result[pos]) for pos in result}

    def move_left(space, filepos, filesize, file_id) -> (int, int, int):
        write_idx = 0
        writepos, free = space[write_idx]
        while writepos < filepos:
            if filesize <= free:
                space[write_idx] = writepos + filesize, free - filesize
                if free == filesize:   # space is filled
                    del space[write_idx]
                return writepos, filesize, file_id
            else:
                write_idx += 1
                writepos, free = space[write_idx]
        return filepos, filesize, file_id

    def move_files(space, files):
        positions = sorted(files, reverse=True)
        return [move_left(space, pos, *files[pos]) for pos in positions]

    def checksum(pos, size, id):
        return id * (2 * pos + size - 1) * size // 2

    part1 = sum(checksum(*file) for file in compress(space, files))
    part2 = sum(checksum(*file) for file in move_files(space, files))

    return part1, part2

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

    # TODO: keep track of all times a certain position was traversed

    part2 = 0
    path = set()
    for (r, c, dr, dc), t in zip(route, count()):
        block = forward(r, c, dr, dc)
        # TODO: was the block traversed before time t?
        if block not in path and world.get(block, '') == '.':
            world[block] = '#'
            for p in trail(r, c, dc, -dr):
                if (p[0], p[1]) not in world:
                    break
                elif visited.get(p, 100000000) < t:
                    part2 += 1
                    break
                # TODO: if the point is visited later on the route, it will not lead to a loop unless
                #       the point can't be reached because it is blocked by the block
                #       so break if
            else:
                part2 += 1    # got stuck in a loop
            world[block] = '.'
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
