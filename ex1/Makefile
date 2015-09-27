.PHONY: ans-1.1

sample-output.txt: sample-mapper.py sample-reducer.py
	rm -f sample-output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./sample-mapper.py | sort | ./sample-reducer.py \
		| tee -a sample-output.txt

1.1-output.txt: 1.1-mapper.py sample-reducer.py
	rm -f 1.1-output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./1.1-mapper.py | sort | ./sample-reducer.py \
		| tee -a 1.1-output.txt

ans-1.1: 1.1-output.txt
	@egrep '^my\t' 1.1-output.txt
	@egrep '^hello\t' 1.1-output.txt || true
	@egrep '^friend\t' 1.1-output.txt
	@egrep '^little\t' 1.1-output.txt
	@egrep '^say\t' 1.1-output.txt
	@egrep '^and\t' 1.1-output.txt
	@egrep '^to\t' 1.1-output.txt
