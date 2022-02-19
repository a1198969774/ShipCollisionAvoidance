import tensorflow as tf
import numpy as np

def create_conv_network(input_frames,trainable):
    conv1_W = tf.get_variable(shape=[8,8,1,16],name='conv1_W',   ## 4-1
                              trainable=trainable,initializer=tf.contrib.layers.xavier_initializer())
    conv1_b = tf.Variable(tf.zeros([16], dtype=tf.float32),
                          name='conv1_b', trainable=trainable)
    conv1 = tf.nn.conv2d(input_frames, conv1_W, strides=[1, 4, 4, 1],
                         padding='VALID', name='conv1')
    # (batch size, 20, 20, 16)
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')
    conv2_W = tf.get_variable(shape=[4, 4, 16, 32], name='conv2_W',
                              trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros([32], dtype=tf.float32), name='conv2_b',
                          dtype=tf.float32, trainable=trainable)
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1],
                         padding='VALID', name='conv2')
    # (batch size, 9, 9, 32)
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')

    flat_output2_size = 16928
    flat_output2 = tf.reshape(output2, [-1, flat_output2_size], name='flat_output2')

    return flat_output2, flat_output2_size, [conv1_W, conv1_b, conv2_W, conv2_b], 1

def create_lstm_network(input_frames,trainable):
    Num_cellState = 256
    x_unstack = tf.unstack(input_frames, axis=2) #按axis切分成 axis個數組
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=Num_cellState)
    rnn_out, rnn_state = tf.nn.static_rnn(
        inputs=x_unstack, cell=cell, dtype=tf.float32)
    flat_output2 = tf.reshape(rnn_out, [-1, len(rnn_out) * Num_cellState], name='flat_output2')
    return flat_output2, len(rnn_out) * Num_cellState, [cell], rnn_out

def create_lstm_conv_network(input_frames,trainable):
    conv1_W = tf.get_variable(shape=[8, 8, 4, 16], name='conv1_W',  ## 4-1
                              trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv1_b = tf.Variable(tf.zeros([16], dtype=tf.float32),
                          name='conv1_b', trainable=trainable)
    conv1 = tf.nn.conv2d(input_frames[1], conv1_W, strides=[1, 4, 4, 1],
                         padding='VALID', name='conv1')
    # (batch size, 20, 20, 16)
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')
    conv2_W = tf.get_variable(shape=[4, 4, 16, 32], name='conv2_W',
                              trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros([32], dtype=tf.float32), name='conv2_b',
                          dtype=tf.float32, trainable=trainable)
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1],
                         padding='VALID', name='conv2')
    # (batch size, 9, 9, 32)
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')


    # flat_output2 = tf.reshape(output2, [-1, flat_output2_size], name='flat_output2')

    Num_cellState = 256
    x_unstack = tf.unstack(input_frames[0], axis=2)  # 按axis切分成 axis個數組
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=Num_cellState)
    rnn_out, rnn_state = tf.nn.static_rnn(
        inputs=x_unstack, cell=cell, dtype=tf.float32)

    h_pool3_flat = tf.reshape(
        output2, [-1, 61 * 61 * 32])  # 将tensor打平到vector中
    rnn_out = rnn_out[-1]

    flat_output2 = tf.concat([h_pool3_flat, rnn_out], axis=1, name='flat_output2')
    # flat_output2 = tf.reshape(h_concat, [-1, len(rnn_out) * Num_cellState + flat_output2_size], name='flat_output2')
    flat_output2_size = 119072 + 256
    return flat_output2, flat_output2_size, [conv1_W, conv1_b, conv2_W, conv2_b, cell], rnn_out

def create_deep_q_network(input_frames,num_actions,create_network_cnn_or_lstm,trainable,noisy):
    flat_output,flat_output_size,parameter_list, rnn_out = create_network_cnn_or_lstm(input_frames,trainable)

    if noisy == False:
        fc1_W = tf.get_variable(shape=[flat_output_size,256],name='fc1_W',
                                trainable=trainable,initializer=tf.contrib.layers.xavier_initializer())

        fc1_b = tf.Variable(tf.zeros([256],dtype=tf.float32),name='fc1_b',
                            trainable=trainable)

        output3 = tf.nn.relu(tf.matmul(flat_output,fc1_W)+fc1_b,name='output3')

        fc2_W = tf.get_variable(shape=[256,num_actions],name='fc2_W',trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())

        fc2_b = tf.Variable(tf.zeros([num_actions],dtype=tf.float32),name='fc2_b',trainable=trainable)

        q_network = tf.nn.relu(tf.matmul(output3,fc2_W) + fc2_b,name='q_network')

        parameter_list += [fc1_W,fc1_b,fc2_W,fc2_b]

    else:

        output3, parameter_list_output3 = noisy_dense(flat_output, name='noisy_fc1',
                                                      input_size=flat_output_size, output_size=256,
                                                      activation_fn=tf.nn.relu, trainable=trainable)
        q_network, parameter_list_q_network = noisy_dense(output3, name='noisy_fc2',
                                                          input_size=256, output_size=num_actions, trainable=trainable)
        parameter_list += parameter_list_output3 + parameter_list_q_network
    return q_network, parameter_list, flat_output,  output3, rnn_out


def create_duel_q_network(input_frames,num_actions,create_network_cnn_or_lstm,trainable,noisy):
    flat_output, flat_output_size, parameter_list = create_network_cnn_or_lstm(input_frames, trainable)

    if noisy == False:
        fcV_W = tf.get_variable(shape=[flat_output_size, 512], name='fcV_W',
                                trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        fcV_b = tf.Variable(tf.zeros([512], dtype=tf.float32), name='fcV_b',
                            dtype=tf.float32, trainable=trainable)
        outputV = tf.nn.relu(tf.matmul(flat_output, fcV_W) + fcV_b, name='outputV')

        fcV2_W = tf.get_variable(shape=[512, 1], name='fcV2_W',
                                 trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        fcV2_b = tf.Variable(tf.zeros([1], dtype=tf.float32), name='fcV2_b',
                             trainable=trainable)
        outputV2 = tf.matmul(outputV, fcV2_W) + fcV2_b # V


        fcA_W = tf.get_variable(shape=[flat_output_size, 512], name='fcA_W',
                                trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        fcA_b = tf.Variable(tf.zeros([512], dtype=tf.float32), name='fcA_b',
                            trainable=trainable)
        outputA = tf.nn.relu(tf.matmul(flat_output, fcA_W) + fcA_b, name='outputA')

        fcA2_W = tf.get_variable(shape=[512, num_actions], name='fcA2_W',
                                 trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        fcA2_b = tf.Variable(tf.zeros([num_actions], dtype=tf.float32), name='fcA2_b',
                             trainable=trainable)
        outputA2 = tf.matmul(outputA, fcA2_W) + fcA2_b # 优势函数

        parameter_list += [fcV_W, fcV_b, fcV2_W, fcV2_b, fcA_W, fcA_b, fcA2_W, fcA2_b]
    else:
        outputV, parameter_list_outputV = noisy_dense(flat_output, name='fcV',
                                                      input_size=flat_output_size, output_size=512, trainable=trainable,
                                                      activation_fn=tf.nn.relu)
        outputV2, parameter_list_outputV2 = noisy_dense(outputV, name='fcV2',
                                                        input_size=512, output_size=1, trainable=trainable)
        ouputA, parameter_list_outputA = noisy_dense(flat_output, name='fcA',
                                                     input_size=flat_output_size, output_size=512, trainable=trainable,
                                                     activation_fn=tf.nn.relu)
        outputA2, parameter_list_outputA2 = noisy_dense(ouputA, name='fcA2',
                                                        input_size=512, output_size=num_actions, trainable=trainable)
        parameter_list += parameter_list_outputA + parameter_list_outputA2 + \
                          parameter_list_outputV + parameter_list_outputV2

    q_network = tf.nn.relu(outputV2 + outputA2 - tf.reduce_mean(outputA2), name='q_network')

    return q_network, parameter_list


def create_model(window, is_cnn, input_shape, input_length, num_actions,model_name,create_network_fn,create_network_cnn_or_lstm,trainable,noisy):
    """创建Q网络"""
    with tf.variable_scope(model_name):
        if is_cnn == 1:
            input_frames = tf.placeholder(tf.float32,[None,input_shape[0],input_shape[1],window],name='input_frames')
            input_frames1 = []
            input_frames2 = []
        elif is_cnn == 0:
            input_frames = tf.placeholder(tf.float32, [None, input_length, window],
                                          name='input_frames')
            input_frames1 = []
            input_frames2 = []
        else:
            input_frames2 = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], window],
                                          name='input_frames2')
            input_frames1 = tf.placeholder(tf.float32, [None, input_length, window],
                                          name='input_frames1')
            input_frames = []
            input_frames.append(input_frames1)
            input_frames.append(input_frames2)
        q_network,parameter_list, out1, out2, rnn_out = create_network_fn(input_frames,num_actions,create_network_cnn_or_lstm,trainable,noisy)
        # tf.reduce_max按行求最值
        mean_max_q = tf.reduce_mean(tf.reduce_max(q_network,axis=[1]),name='mean_max_q')
        action = tf.argmax(q_network,axis=1)  #返回最大值的索引

        model = {
            'q_values':q_network,
            'input_frames':input_frames,
            'input_frames1': input_frames1,
            'input_frames2': input_frames2,
            'mean_max_q':mean_max_q,
            'action':action,
            'out1': out1,
            'out2': out2,
            'rnn_out': rnn_out,
        }

    return model,parameter_list


def create_distributional_model(window, is_cnn, input_shape, input_length, num_actions,model_name,create_network_fn,create_network_cnn_or_lstm,trainable,noisy):
    N_atoms = 51
    V_Max = 20.0
    V_Min = 0.0
    Delta_z = (V_Max - V_Min) / (N_atoms - 1)
    z_list = tf.constant([V_Min + i * Delta_z for i in range(N_atoms)],dtype=tf.float32)
    z_list_broadcasted = tf.tile(tf.reshape(z_list,[1,N_atoms]),[num_actions,1]) # batch * num_actions * N_atoms

    with tf.variable_scope(model_name):
        if is_cnn == 1:
            input_frames = tf.placeholder(tf.float32,[None,input_shape[0],input_shape[1],window],name='input_frames')
        elif is_cnn == 0:
            input_frames = tf.placeholder(tf.float32, [None, input_length, window],
                                          name='input_frames')
        else:
            input_frames2 = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], window],
                                           name='input_frames2')
            input_frames1 = tf.placeholder(tf.float32, [None, input_length, window],
                                           name='input_frames1')
            input_frames = []
            input_frames.append(input_frames1)
            input_frames.append(input_frames2)
        q_distributional_network,parameter_list = create_network_fn(input_frames,num_actions * N_atoms,create_network_cnn_or_lstm,trainable,noisy)

        q_distributional_network = tf.reshape(q_distributional_network,[-1,num_actions,N_atoms])

        q_distributional_network = tf.nn.softmax(q_distributional_network,dim=2)
        # 防止NAN
        q_distributional_network = tf.clip_by_value(q_distributional_network, 1e-8, 1.0 - 1e-8)
        q_network = tf.multiply(q_distributional_network ,z_list_broadcasted)
        q_network = tf.reduce_sum(q_network,axis=2,name='q_values')

        mean_max_q = tf.reduce_mean(tf.reduce_max(q_network,axis=[1]),name='mean_max_q')
        action = tf.argmax(q_network,axis=1)

        model = {
            'q_distributional_network' : q_distributional_network,
            'q_values':q_network,
            'input_frames':input_frames,
            'mean_max_q':mean_max_q,
            'action':action
        }

    return model,parameter_list


def noisy_dense(x,input_size,output_size,name,trainable,activation_fn=tf.identity):

    def f(x):
        return tf.multiply(tf.sign(x),tf.pow(tf.abs(x),0.5))

    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(input_size, 0.5),
                                                maxval=1*1/np.power(input_size, 0.5))

    sigma_init = tf.constant_initializer(0.4 / np.power(input_size,0.5))

    p = tf.random_normal([input_size,1])
    q = tf.random_normal([1,output_size])

    f_p = f(p)
    f_q = f(q)

    w_epsilon = f_p * f_q
    b_epsilon = tf.squeeze(f_q)

    w_mu = tf.get_variable(name + "/w_mu", [input_size, output_size],
                           initializer=mu_init, trainable=trainable)
    w_sigma = tf.get_variable(name + "/w_sigma", [input_size, output_size],
                              initializer=sigma_init, trainable=trainable)

    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x,w)

    b_mu = tf.get_variable(name + "/b_mu", [output_size],
                           initializer=mu_init, trainable=trainable)
    b_sigma = tf.get_variable(name + "/b_sigma", [output_size],
                              initializer=sigma_init, trainable=trainable)
    b = b_mu + tf.multiply(b_sigma, b_epsilon)
    return activation_fn(ret + b), [w_mu, w_sigma, b_mu, b_sigma]



