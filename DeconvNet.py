import os
import random
import tensorflow as tf
import time
import numpy as np
import cv2


class DeconvNet:
    def __init__(self, use_cpu=False, checkpoint_dir='./checkpoints/'):
        

        self.build(use_cpu=use_cpu)

        self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        config = tf.ConfigProto(allow_soft_placement = True)
        self.session = tf.Session(config = config)
        self.image = tf.placeholder(tf.float32, shape=(1,224,224,3))        
        self.session.run(tf.initialize_all_variables())

        
    
    def train(self, train_stage=1, training_steps=3, restore_session=False, learning_rate=1e-4):
        if restore_session:
            step_start = restore_session()
        else:
            step_start = 0

        if train_stage == 1:
            trainset = open('data/stage_1_train_imgset/train.txt').readlines()
        else:
            trainset = open('data/stage_2_train_imgset/train.txt').readlines()
        print('trainset size is.............',len(trainset))
        for i in range(step_start, step_start+training_steps):
            # pick random line from file
            #print('curr image....',trainset[i])
            #random_line = random.choice(trainset)
            random_line = trainset[i]
            #print('random line is ..................',random_line)
            image_file = random_line.split(' ')[0]
            ground_truth_file = random_line.split(' ')[1]
            #print('ground truth file is...........',ground_truth_file[:-1])
            image1 = (cv2.imread('data' + image_file))
            image = np.float32(cv2.resize(image1,(224, 224), interpolation = cv2.INTER_CUBIC))

            ground_truth1 = cv2.imread('data' + ground_truth_file[:-1], cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.resize(ground_truth1,(224, 224), interpolation = cv2.INTER_CUBIC)
            
            # norm to 21 classes [0-20] (see paper)
            ground_truth = (ground_truth / 255) * 20
            print('run train step: '+str(i))
            start = time.time()
            self.train_step.run(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth], self.rate: learning_rate})

            if i % 499 == 0 or i == (step_start+training_steps-1):
                print('step {} finished in {:.2f} s with loss of {:.6f}'.format(
                    i, time.time() - start, self.loss.eval(session=self.session, feed_dict={self.x: [image], self.y: [ground_truth]})))
            if i == (step_start+training_steps-1):    
                self.saver.save(self.session, self.checkpoint_dir+'model', global_step=i)
                print('Model {} saved'.format(i))
            
            

    def build(self, use_cpu=False):
      
        #writer = tf.train.SummaryWriter("./log_tb")

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.device(device):
            self.x = tf.placeholder(tf.float32, shape=(1, 224,224, 3))
            self.y = tf.placeholder(tf.int64, shape=(1, 224, 224))
            self.image = tf.placeholder(tf.float32, shape=(1,224,224,3))
            expected = tf.expand_dims(self.y, -1)
            self.rate = tf.placeholder(tf.float32, shape=[])

            conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv_1_1')
            conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')
            img_size = 224
            channels = 64
           
            print('conv1 is ................',conv_1_2)
            V = tf.slice(conv_1_2, (0, 0, 0, 0), (1, -1, -1, -1), name='slice_first_input')
            V = tf.reshape(V, (img_size, img_size, channels))

            # Reorder so the channels are in the first dimension, x and y follow.
            V = tf.transpose(V, (2, 0, 1))
            # Bring into shape expected by image_summary
            V = tf.reshape(V, (-1, img_size, img_size, 1))
            summ = tf.image_summary("first _conv",V)
            #with tf.Session() as sess2:
            #sumary = self.session.run(summ)
            #writer = tf.train.SummaryWriter('./logs')
            #writer.add_summary(sumary)
            #writer.close()

            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

            conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
            conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)
            conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
            conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
            conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

            pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

            conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
            conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
            conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

            pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

            conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
            conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
            conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

            pool_5, pool_5_argmax = self.pool_layer(conv_5_3)
            print('pool5 is.........',pool_5)
            fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
            fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')
            print('fv7 is ------------',fc_7)
            deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv', 1)
            print('deconv6 is..............',deconv_fc_6)
            print('conv_5_3 shape is........',tf.shape(conv_5_3))
            #unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))
            unpool_5 = self.unpool(pool_5_argmax)
            print('unpool_5............',unpool_5)
            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3', 2)
            deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2', 2)
            deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1', 2)

            #unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))
            unpool_4 = self.unpool(pool_4_argmax)
            print('unpool_4............',unpool_4)

            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3', 3)
            deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2', 3)
            deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1', 4)

            #unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))

            unpool_3 = self.unpool(pool_3_argmax)
            print('unpool_3............',unpool_3)


            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3', 5)
            deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2',5)
            deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1',6)

            #unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))
            unpool_2 = self.unpool(pool_2_argmax)
            print('unpool_2............',unpool_2)

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2', 7)
            deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1', 8)

            #unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))
            unpool_1 = self.unpool(pool_1_argmax)
            print('unpool_1............',unpool_1)
	    deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2', 9)
            deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1', 10)

            score_1 = self.deconv_layer(deconv_1_1, [1, 1, 21, 32], 21, 'score_1', 11)
            print('score_1..................',score_1)


            logits = tf.reshape(score_1, (-1, 21))
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(expected, [-1]), name='x_entropy')
            # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

            self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))

            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(score_1, ground_truth, name='Cross_Entropy')
            #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            # tf.add_to_collection('losses', cross_entropy_mean)

            # self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            # self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
 
            # self.prediction = tf.argmax(score_1, dimension=3)
            # self.accuracy = tf.reduce_sum(tf.pow(self.prediction - expected, 2))


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)

    def pool_layer(self, x):
        
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def deconv_layer(self, x, W_shape, b_shape, name, a, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        print('x is ',x)
        x_shape = tf.shape(x)
        print('x_shape is   ',x_shape)
        if a is 1:
            out_shape1 = tf.constant([1,7,7,512])
            return tf.nn.conv2d_transpose(x, W, out_shape1, [1, 1, 1, 1], padding=padding) + b
        if a is 2:
            out_shape2 = tf.constant([1,14,14,512])
            return tf.nn.conv2d_transpose(x, W, out_shape2, [1, 1, 1, 1], padding=padding) + b
        if a is 3:
            out_shape3 = tf.constant([1,28,28,512])
            return tf.nn.conv2d_transpose(x, W, out_shape3, [1, 1, 1, 1], padding=padding) + b
        if a is 4:
            out_shape4 = tf.constant([1,28,28,256])
            return tf.nn.conv2d_transpose(x, W, out_shape4, [1, 1, 1, 1], padding=padding) + b
        if a is 5:
            out_shape5 = tf.constant([1,56,56,256])
            return tf.nn.conv2d_transpose(x, W, out_shape5, [1, 1, 1, 1], padding=padding) + b
        if a is 6:
            out_shape6 = tf.constant([1,56,56,128])
            return tf.nn.conv2d_transpose(x, W, out_shape6, [1, 1, 1, 1], padding=padding) + b
        if a is 7:
            out_shape7 = tf.constant([1,112,112,128])
            return tf.nn.conv2d_transpose(x, W, out_shape7, [1, 1, 1, 1], padding=padding) + b
        if a is 8:
            out_shape8 = tf.constant([1,112,112,64])
            return tf.nn.conv2d_transpose(x, W, out_shape8, [1, 1, 1, 1], padding=padding) + b
        if a is 9:
            out_shape9 = tf.constant([1,224,224,64])
            return tf.nn.conv2d_transpose(x, W, out_shape9, [1, 1, 1, 1], padding=padding) + b
        if a is 10:
            out_shape10 = tf.constant([1,224,224,32])
            return tf.nn.conv2d_transpose(x, W, out_shape10, [1, 1, 1, 1], padding=padding) + b
        if a is 11:
            out_shape11 = tf.constant([1,224,224,21])
            return tf.nn.conv2d_transpose(x, W, out_shape11, [1, 1, 1, 1], padding=padding) + b

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.pack(output_list)

    def unpool(self, value, name='unpool'):
        with tf.name_scope(name) as scope:
            sh = value.get_shape().as_list()
            dim = len(sh[1:-1])
            out = (tf.reshape(value, [-1] + sh[-dim:]))
            for i in range(dim, 0, -1):
                out = tf.concat(i, [out, tf.zeros_like(out)])
            out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
            out = tf.reshape(out, out_size, name=scope)
        return tf.to_float(out, name='ToFloat')

		

