#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import sys
import re
import string

alnum = re.compile(r'[\W_]+', re.UNICODE)
atoz = re.compile(r'[^a-z]+')

for line in sys.stdin:
    line = alnum.sub(' ', line.strip())
    words = line.split()
    for word in words:
        word = word.lower()
        if len(atoz.sub('', word)) == len(word):
            print('%s\t%s' % (word, 1))
