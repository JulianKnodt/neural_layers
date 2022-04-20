setup:
	sudo -E python3 setup.py install

mnist:
	python3 runner.py mnist
vit:
	python3 runner.py vit
pointnet:
	python3 runner.py pointnet
