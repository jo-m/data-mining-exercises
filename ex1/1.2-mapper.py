#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import re
import string

A, B = 10, 35

for line in sys.stdin:
    word, _, count = line.strip().partition('\t')
    count = int(count)
    if count >= A and count <= B:
        print('%c %06d %s' % (word[0], count, word))
