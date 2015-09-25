.PHONY: run
run:
	rm -f output.txt
	cat hw1-data/* | iconv -f iso-8859-1 -t UTF-8 \
		| ./mapper.py | sort | ./reducer.py \
		tee -a output.txt
