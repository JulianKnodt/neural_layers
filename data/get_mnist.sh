curl "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" --output train_image_mnist.gz
curl "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" --output train_label_mnist.gz
gunzip train_image_mnist.gz
gunzip train_label_mnist.gz

curl "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" --output t10k_image_mnist.gz
curl "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" --output t10k_label_mnist.gz
gunzip t10k_image_mnist.gz
gunzip t10k_label_mnist.gz
