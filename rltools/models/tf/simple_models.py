import prettytensor as pt
import tensorflow as tf

def softmax_mlp(input_data, output_size, layers=[32,32], activation=tf.nn.relu):

    net = pt.wrap(input_data).sequential()
    net.flatten()
    for l_size in layers:
        net.fully_connected(l_size, activation_fn=activation) 
    output_layer, _ = net.softmax_classifier(output_size)

    return output_layer
