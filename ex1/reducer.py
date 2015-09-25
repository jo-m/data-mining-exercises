#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

for line in sys.stdin:
    line = line.strip()

    word, count = line.split('\t', 1)
    # initial count
    count = int(count)

    if current_word == word:
        # counting on an already occured word
        current_count += count
    else:
        # new word occuring, init new
        if current_word:
            print('%s\t%s' % (current_word, current_count))
        current_count = count
        current_word = word

if current_word == word:
    print('%s\t%s' % (current_word, current_count))
