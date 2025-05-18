"""
comprehensive_sample.py

A deliberately feature-rich script for compilation or byte-code
experiments.  It touches on most built-in data types, control-flow
paths, the random & math modules, comprehensions, generators, context
managers, exceptions, threading, futures, and file-I/O.

Run directly:     python comprehensive_sample.py
"""
from __future__ import annotations
import itertools
import math
import random
import threading
from collections import Counter, deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median

integer_var: int = 42
float_var: float = math.e
complex_var: complex = 2+3j
bool_var: bool = True
str_var: str = 'Hello, Python!'
bytes_var: bytes = b'\xde\xad\xbe\xef'
bytearray_var: bytearray = bytearray(b'mutable')

list_var: list[int] = [random.randint(1, 100) for _ in range(10)]
tuple_var: tuple[int, ...] = tuple(list_var)
set_var: set[int] = set(list_var)
frozenset_var: frozenset[int] = frozenset(set_var)
dict_var: dict[str, int] = {f'idx_{i}': n for i, n in enumerate(list_var)}

Point = namedtuple('Point', ['x', 'y'])

@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def __repr__(self) -> str:
        return f'Vector2D(x={self.x}, y={self.y})'

def estimate_pi(samples: int = 1000000) -> float:
    inside = 0
    for _ in range(samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / samples

def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def stats_on(numbers: list[int]) -> dict[str, float]:
    nums = sorted(numbers)
    return {
        'min': nums[0],
        'max': nums[-1],
        'mean': mean(nums),
        'median': median(nums),
        'unique': len(set(nums)),
    }

def _worker(seed: int) -> float:
    random.seed(seed)
    return estimate_pi(100000)

def parallel_pi(workers: int = 4) -> float:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        estimates = list(ex.map(_worker, range(workers)))
    return mean(estimates)

_counter_lock = threading.Lock()
shared_counter = 0

def increment(times: int):
    global shared_counter
    for _ in range(times):
        with _counter_lock:
            shared_counter += 1

def threaded_increment(total: int = 1000, workers: int = 5) -> int:
    global shared_counter
    threads = [threading.Thread(target=increment, args=(total // workers,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return shared_counter

def primes(limit: int):
    for n in itertools.islice(itertools.count(2), limit):
        if all(n % p for p in range(2, int(math.sqrt(n)) + 1)):
            yield n

def write_report(path: str, text: str):
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
    except OSError as exc:
        print('Write error:', exc)

def main():
    print('######################################################################')
    print('Comprehensive Sample Program —', datetime.now().isoformat())
    print('######################################################################')

    # Control flow
    if bool_var and integer_var > 0:
        print('integer_var is positive')
    elif integer_var == 0:
        print('integer_var is zero')
    else:
        print('integer_var is negative')

    # Loop
    c = 3
    while c:
        print('Countdown:', c)
        c -= 1

    # Data structures and functions
    stats = stats_on(list_var)
    print('Random list stats:', stats)

    print('Fibonacci(10):', fibonacci(10))

    print('First 10 primes:', list(itertools.islice(primes(30), 10)))

    # Class usage
    v1, v2 = Vector2D(3, 4), Vector2D(-2, 5)
    v3 = v1 + v2
    print('Vector add:', v3, '|v3| =', v3.magnitude())

    # Collections
    print('Letter frequencies:', dict(Counter('abracadabra')))

    # Map and Lambda
    print('Squares 0-4:', list(map(lambda x: x * x, range(5))))

    # Threading/Futures
    print('Average π estimate (threads):', parallel_pi())

    # Global state with threading
    print('Final shared_counter:', threaded_increment())

    # Bytes and Bytearray
    print('bytes_var hex:', bytes_var.hex())
    bytearray_var.extend(b' data')
    print('bytearray_var:', bytearray_var.decode())

    # Dictionary comprehension usage
    doubled = {k: v * 2 for k, v in dict_var.items()}
    print('Dict doubled values:', doubled)

    # Deque usage
    dq = deque(range(5))
    dq.rotate(2)
    print('Rotated deque:', list(dq))

    # File I/O and Exception Handling
    report = f"""Stats   : {stats}
Vector  : {v3}
π est.  : {parallel_pi():.6f}
Created : {datetime.now()}
"""
    write_report('sample_report.txt', report)
    print('Wrote sample_report.txt')

    print('######################################################################')
    print('Program finished successfully.')

if __name__ == "__main__":
    main()