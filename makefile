setup:
	sudo -E python3 setup.py install
mnist:
	python3 runner.py mnist
coco:
	python3 runner.py coco
