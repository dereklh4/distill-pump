import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def fc_layer(inp, size, name):
    W_layer = tf.Variable(tf.truncated_normal([inp.get_shape()[1].value, size],stddev=0.1))
    b_layer = tf.Variable(tf.constant(0, tf.float32, shape=[size]))
    res = tf.matmul(inp, W_layer) + b_layer
    tf.histogram_summary(name + "_weights", W_layer)
    return res, [W_layer, b_layer]

def fully_connected(name):
    L1, fc_params1 = fc_layer(tf.reshape(X, [-1,784]) , 100, name + "L1")
    L1a = tf.nn.sigmoid(L1)
    res, fc_params2 = fc_layer(L1a , 10, name + "L2")
    return tf.nn.softmax((1.0/T) * res), fc_params1 + fc_params2

def prec(y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1)), tf.float32))

# distillate knowledge from donor to acceptor
def distillate(net_name):
    acceptor_prec = prec(y_acceptor)
    donor_prec = prec(y_donor)

    distil_writer = tf.train.SummaryWriter("logs/" + net_name + "/train", flush_secs=5)
    distil_test_writer = tf.train.SummaryWriter("logs/" + net_name + "/test", flush_secs=5)

    tf.scalar_summary('distill_accuracy', acceptor_prec)
    summaries = tf.merge_all_summaries()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # Donor loading
        print(donor_params)
        donor_saver = tf.train.Saver(donor_params)
        donor_saver.restore(sess, 'checkpoints/' + donor_name)

        acc_saver = tf.train.Saver(acceptor_params)

        k = 1
        for i in range(1):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                val_prec, val_donor_prec, log_summaries, val_distil_ent, val_truth_ent, _ = sess.run(
                    [acceptor_prec, donor_prec, summaries, distill_ent, truth_ent, distil_step],
                    feed_dict={X: trX[start:end],
                               Y: trY[start:end],
                               p_keep_conv: 1,
                               p_keep_hidden: 1 })

                test_val_prec, test_log_summaries = sess.run([acceptor_prec, summaries],
                                                             feed_dict={X: teX[:test_size],
                                                                        Y: teY[:test_size],
                                                                        p_keep_conv: 1.0,
                                                                        p_keep_hidden: 1.0})

                distil_writer.add_summary(log_summaries, k)
                distil_test_writer.add_summary(test_log_summaries, k)
                print(i, k, 'distillation distil_ent:', val_distil_ent, 'truth_ent', val_truth_ent, 'donor_prec: ', val_donor_prec, '; train_prec', val_prec, '; test_prec', test_val_prec)

                k = k + 1

            acc_saver.save(sess, "checkpoints/" + net_name, global_step = i)

# train net using back-prop
def train_net(train_step, net_prec, net_params, net_name):
    net_prec = prec(y_acceptor)

    writer = tf.train.SummaryWriter("logs/" + net_name + "/train", flush_secs=5)
    test_writer = tf.train.SummaryWriter("logs/" + net_name + "/test", flush_secs=5)

    tf.scalar_summary('train_accuracy', net_prec)
    summaries = tf.merge_all_summaries()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        net_saver = tf.train.Saver(net_params)

        k = 1
        for i in range(1):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                val_prec, log_summaries, _ = sess.run([net_prec, summaries, train_step],
                                                      feed_dict={X: trX[start:end],
                                                                 Y: trY[start:end],
                                                                 p_keep_conv: 1,
                                                                 p_keep_hidden: 1 })

                test_val_prec, test_log_summaries = sess.run([net_prec, summaries],
                                                             feed_dict={X: teX[:test_size],
                                                                        Y: teY[:test_size],
                                                                        p_keep_conv: 1,
                                                                        p_keep_hidden: 1 })

                writer.add_summary(log_summaries, k)
                test_writer.add_summary(test_log_summaries, k)
                print(i, k, 'train_prec', val_prec, '; test_prec', test_val_prec)
                k = k + 1
            net_saver.save(sess, "checkpoints/" + net_name, global_step = i)


if __name__ == "__main__":

    batch_size = 128
    test_size = 1000

    donor_name = "l2_full-1000"
    acceptor_name = "distil_comb_L3"
    train_output_name = "L4"

    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    T = 1

    # donor & acceptor networks

    y_donor, donor_params = fully_connected(donor_name)
    y_acceptor, acceptor_params = fully_connected(acceptor_name)

    # distillation
    distill_ent = tf.reduce_mean( - tf.reduce_sum(tf.stop_gradient(y_donor) * tf.log(y_acceptor), reduction_indices=1))
    truth_ent = tf.reduce_mean( - tf.reduce_sum(Y * tf.log(y_acceptor), reduction_indices=1))

    distil_step = tf.train.GradientDescentOptimizer(0.1).minimize(distill_ent + truth_ent)

    # MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
    teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

    train_cross_ent = tf.reduce_mean( - tf.reduce_sum(Y * tf.log(y_acceptor), reduction_indices=1))
    train_prec = prec(y_acceptor)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(train_cross_ent)

    distillate(acceptor_name)
    train_net(train_step, train_prec, acceptor_params, train_output_name)