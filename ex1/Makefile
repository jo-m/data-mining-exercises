.PHONY:

sample-output.txt: sample-mapper.py sample-reducer.py
	rm -f output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./sample-mapper.py | sort | ./sample-reducer.py \
		tee -a sample-output.txt

