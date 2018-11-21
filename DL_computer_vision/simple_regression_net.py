    def train(self, iterations, epochs, batch_size):

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        image = tf.placeholder(tf.float32, shape=[None, 299, 299, 2], name='image')
        pred_x = tf.placeholder(tf.float32, shape=[None, len(self.pred_x[1])], name='pred_x')
        pred_y = tf.placeholder(tf.float32, shape=[None, len(self.pred_y[1])], name='pred_y')
        gt_x = tf.placeholder(tf.float32, shape=[None, len(self.gt_x[1])], name='gt_x')
        gt_y = tf.placeholder(tf.float32, shape=[None, len(self.gt_x[1])], name='gt_y')
        print(pred_x, pred_y)

        # layer 1
        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        x_image = tf.reshape(image, [-1, 299, 299, 1])
        h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1))
        h_pool1 = max_pool_2x2(h_conv1)
        print(h_pool1)

        # layer2
        W_conv2 = weight_variable([3, 3, 16, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2))
        h_pool2 = max_pool_2x2(h_conv2)
        print(h_pool2)

        # layer3
        W_conv3 = weight_variable([3, 3, 48, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, W_conv3) + b_conv3))
        h_pool3 = max_pool_2x2(h_conv3)
        print(h_pool3)

        # layer4
        W_conv4 = weight_variable([2, 2, 64, 64])
        b_conv4 = bias_variable([64])
        h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, W_conv4) + b_conv4))
        h_pool4 = h_conv4
        print(h_pool4)

        #  layer5
        W_fc1 = weight_variable([2 * 2 * 64, 100])
        b_fc1 = bias_variable([100])
        h_pool4_flat = tf.layers.Flatten()(h_pool4)
        h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, W_fc1) + b_fc1))
        keep_prob = tf.placeholder_with_default(0.0, [], 'dropout')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1_drop)

        # landmark
        W_fc_landmark_x0 = weight_variable([100, 50])
        b_fc_landmark_x0 = bias_variable([50])
        x_landmark = tf.matmul(h_fc1_drop, W_fc_landmark_x0) + b_fc_landmark_x0

        W_fc_landmark_y0 = weight_variable([100, 50])
        b_fc_landmark_y0 = bias_variable([50])
        y_landmark = tf.matmul(h_fc1_drop, W_fc_landmark_y0) + b_fc_landmark_y0

        x_landmark = tf.reshape(x_landmark, (-1, 100))
        x_features2 = tf.concat((x_landmark, pred_x), -1)
        W_fc_landmark_x = weight_variable([150, OUTPUT_LEN])
        b_fc_landmark_x = bias_variable([OUTPUT_LEN])
        x_landmark = tf.matmul(x_features2, W_fc_landmark_x) + b_fc_landmark_x

        y_landmark = tf.reshape(y_landmark, (-1, 100))
        y_features2 = tf.concat((y_landmark, pred_y), -1)
        W_fc_landmark_y = weight_variable([150, OUTPUT_LEN])
        b_fc_landmark_y = bias_variable([OUTPUT_LEN])
        y_landmark = tf.matmul(y_features2, W_fc_landmark_y) + b_fc_landmark_y

        # cap_min = (10 - 400) / 400
        # y_landmark = tf.map_fn(lambda x: tf.map_fn(lambda y: tf.where(y < cap_min, -1.0, y), x), y_landmark)
        # x_landmark = tf.map_fn(lambda x: tf.map_fn(lambda y: tf.where(y < cap_min, -1.0, y), x), x_landmark)

        tf.add(tf.multiply(400.0, y_landmark), 400.0, name="y_landmark")
        tf.add(tf.multiply(400.0, x_landmark), 400.0, name="x_landmark")

        error = 1 / 2 * tf.reduce_sum(tf.square((gt_x - x_landmark))) + \
                2 * tf.nn.l2_loss(W_fc_landmark_x0) + \
                1 / 2 * tf.reduce_sum(tf.square((gt_y - y_landmark))) + \
                2 * tf.nn.l2_loss(W_fc_landmark_y0)
