import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np

class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = tf.compat.v1.nn.rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.compat.v1.nn.rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = tfa.rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell,
                                          input_keep_prob=args.input_keep_prob,
                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)

        self.cell = cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.input_data = tf.compat.v1.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.compat.v1.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.compat.v1.variable_scope('rnnlm'):
            softmax_w = tf.compat.v1.get_variable("softmax_w",
                                        [args.rnn_size, args.vocab_size])
            softmax_b = tf.compat.v1.get_variable("softmax_b", [args.vocab_size])

        embedding = tf.compat.v1.get_variable("embedding", [args.vocab_size, args.rnn_size])
        inputs = tf.nn.embedding_lookup(params=embedding, ids=self.input_data)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs = tf.nn.dropout(inputs, rate=1 - (args.output_keep_prob))

        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(input=prev, axis=1))
            return tf.nn.embedding_lookup(params=embedding, ids=prev_symbol)

        outputs, last_state = tfa.seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])


        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tfa.seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.compat.v1.name_scope('cost'):
            self.cost = tf.reduce_sum(input_tensor=loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ys=self.cost, xs=tvars),
                args.grad_clip)
        with tf.compat.v1.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.compat.v1.summary.histogram('logits', self.logits)
        tf.compat.v1.summary.histogram('loss', loss)
        tf.compat.v1.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
