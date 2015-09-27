#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from operator import itemgetter
import sys

current_char = None
current_count = 0
skip_char = None

for line in sys.stdin:
    char, count, word = line.strip().split()
    count = int(count)

    if char == skip_char:
        continue

    if current_count >= 20:
        skip_char = current_char
        current_char = None
        current_count = 0
        continue

    if current_char is None:
        current_char = char

    current_count += 1

    print('%s\t%s' % (word, count))
