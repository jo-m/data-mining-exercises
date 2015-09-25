.PHONY: ans-1.1

sample-output.txt: sample-mapper.py sample-reducer.py
	rm -f sample-output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./sample-mapper.py | sort | ./sample-reducer.py \
		tee -a sample-output.txt

1.1-output.txt: 1.1-mapper.py sample-reducer.py
	rm -f 1.1-output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./1.1-mapper.py | sort | ./sample-reducer.py \
		tee -a 1.1-output.txt

ans-1.1: 1.1-output.txt
	grep my 1.1-output.txt
	grep hello 1.1-output.txt
	grep friend 1.1-output.txt
	grep little 1.1-output.txt
	grep say 1.1-output.txt
	grep and 1.1-output.txt
	grep to 1.1-output.txt
