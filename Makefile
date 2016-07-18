init:
	pip install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
	pip install -r requirements.txt

test:
	python -m pytest tests
