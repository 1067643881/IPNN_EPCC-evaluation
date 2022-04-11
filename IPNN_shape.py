result_loss = []
for ipnn_shape in [
    [16, 1],
    [32, 1],
    [64, 1],
    [32, 16, 1],
    [64, 32, 1],
    [128, 64, 1],
    [64, 32, 16, 1],
    [128, 64, 32, 1],
    [256, 128, 64, 1]]:
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from datetime import datetime, date, timedelta
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Layer, Embedding

    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    ################################################ Perpare Data ################################################
    # load data
    df = pd.read_excel('data.xlsx')
    df = df.sample(frac=1)
    npy = np.array(df)

    # city mapping
    city_list = np.unique(df['京津冀城市'].values)
    for i in range(len(npy[:, 0])):
        npy[:, 0][i] = np.where(city_list == npy[:, 0][i])[0][0]

    # XY split
    X = npy[:, :-1]
    Y = npy[:, -1]

    # train/test split
    X_train = X[:285, :]
    Y_train = Y[:285].reshape(-1, 1)

    X_test = X[-15:, :]
    Y_test = Y[-15:].reshape(-1, 1)


    class LookupLayer(Layer):
        def __init__(self, vocabulary_list, vocabulary_type, num_oov_buckets, **kwargs):
            self.vocabulary_list = vocabulary_list
            self.vocabulary_type = vocabulary_type
            self.num_oov_buckets = num_oov_buckets
            assert self.num_oov_buckets > 0
            self.core = None
            super(LookupLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            kv_tensor_initializer = tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.vocabulary_list, dtype=self.vocabulary_type),
                values=tf.range(len(self.vocabulary_list), dtype=tf.int64))
            self.core = tf.lookup.StaticVocabularyTable(num_oov_buckets=max(self.num_oov_buckets, 1),
                                                        initializer=kv_tensor_initializer)
            super(LookupLayer, self).build(input_shape)

        def call(self, input_x, **kwargs):
            return self.core.lookup(input_x)

        def get_config(self):
            config = {'vocabulary_list': self.vocabulary_list,
                      'vocabulary_type': self.vocabulary_type,
                      'num_oov_buckets': self.num_oov_buckets}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))


    class FMLayer(Layer):
        def __init__(self, keepdims=True, **kwargs):
            super(FMLayer, self).__init__(**kwargs)
            self.keepdims = keepdims

        def build(self, input_shape):
            super(FMLayer, self).build(input_shape)

        def call(self, input_x, **kwargs):
            pow_of_sum = tf.pow(tf.reduce_sum(input_x, axis=1), 2)
            sum_of_pow = tf.reduce_sum(tf.pow(input_x, 2), axis=1)
            output = 0.5 * tf.reduce_sum(pow_of_sum - sum_of_pow, axis=1, keepdims=self.keepdims)
            return output


    class IPLayer(Layer):
        def __init__(self, keepdims=True, diagonal=False, **kwargs):
            super(IPLayer, self).__init__(**kwargs)
            self.keepdims = keepdims
            self.diagonal = diagonal
            self.output_indices = list()

        def build(self, input_shape):
            super(IPLayer, self).build(input_shape)
            c = 0
            for i in range(input_shape[1]):
                for j in range(input_shape[1]):
                    if i < j + self.diagonal:
                        self.output_indices.append(c)
                    c += 1
            self.output_indices = tf.constant(self.output_indices, dtype=tf.int32)

        def call(self, input_x, **kwargs):
            input_x_T = tf.transpose(input_x, perm=[0, 2, 1])
            matmul = tf.reshape(tf.matmul(input_x, input_x_T), (-1, input_x.shape[1] ** 2))
            output = tf.gather(matmul, self.output_indices, axis=1)
            return output


    class DNNLayer(Layer):
        def __init__(self, unit_list, activation_list, bias_list, kernel_regularizer_list, **kwargs):
            assert len(unit_list) == len(activation_list) == len(bias_list) == len(kernel_regularizer_list)
            self.unit_list = unit_list
            self.activation_list = activation_list
            self.bias_list = bias_list
            self.kernel_regularizer_list = kernel_regularizer_list
            self.core = None
            super(DNNLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.core = Sequential()
            for i in range(len(self.unit_list)):
                self.core.add(tf.keras.layers.Dense(units=self.unit_list[i],
                                                    activation=self.activation_list[i],
                                                    use_bias=self.bias_list[i],
                                                    kernel_regularizer=self.kernel_regularizer_list[i]))
            super(DNNLayer, self).build(input_shape)

        def call(self, input_x, **kwargs):
            return self.core(input_x)

        def get_config(self, ):
            config = {'unit_list': self.unit_list,
                      'activation_list': self.activation_list,
                      'kernel_regularizer_list': self.kernel_regularizer_list}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))


    class DeepFM():
        def __init__(self, cls_num):
            self.cls_num = cls_num
            self.lr = LRLayer()
            self.fm = FMLayer()
            self.dnn = DNNLayer(unit_list=[512, 256, 1],
                                activation_list=['relu', 'relu', None],
                                bias_list=[True, True, True],
                                kernel_regularizer_list=[None, None, None])
            if self.cls_num == 1:
                self.output_layer = tf.keras.layers.Dense(units=self.cls_num, activation="sigmoid",
                                                          kernel_regularizer=None)
            else:
                self.output_layer = tf.keras.layers.Dense(units=self.cls_num, activation="softmax",
                                                          kernel_regularizer=None)

        def calculate(self, embeddings):
            sf_embedding, df_embedding_split_by_field, df_embedding = embeddings
            lr_output = self.lr(sf_embedding)
            fm_output = self.fm(df_embedding_split_by_field)
            dnn_output = self.dnn(df_embedding)
            output_concat = tf.keras.layers.concatenate([lr_output, fm_output, dnn_output], axis=1)
            model_output = self.output_layer(output_concat)
            return model_output


    class IPNN():
        def __init__(self, cls_num):
            self.cls_num = cls_num
            self.dnn = DNNLayer(unit_list=[128, 64, self.cls_num],
                                activation_list=['relu', 'relu', 'sigmoid' if self.cls_num == 1 else 'softmax'],
                                bias_list=[True, True, True],
                                kernel_regularizer_list=[None, None, None])

        def calculate(self, ip_embedding):
            model_output = self.dnn(ip_embedding)
            return model_output


    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Model
            model_out = model(x_batch, training=True)
            # Loss Value
            loss_value_list = list()
            loss_value_list.append(MSE_loss_fn(tf.reshape(y_batch, (-1, 1)), tf.reshape(model_out, (-1, 1))))
            loss_value = sum(loss_value_list) + sum(model.losses)
            loss_value = tf.nn.compute_average_loss(loss_value, global_batch_size=global_batch_size)
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads = [tf.clip_by_value(grad, -1, 1) for grad in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return model_out, loss_value


    @tf.function
    def test_step(x_batch, y_batch):
        # Model
        model_out = model(x_batch, training=False)
        # Loss Value
        loss_value_list = list()
        loss_value_list.append(MSE_loss_fn(tf.reshape(y_batch, (-1, 1)), tf.reshape(model_out, (-1, 1))))
        loss_value = sum(loss_value_list) + sum(model.losses)
        loss_value = tf.nn.compute_average_loss(loss_value, global_batch_size=global_batch_size)
        return model_out, loss_value


    def info(mode, model, dataset, show):
        # 推理
        Y_true = list()
        Y_pred = list()
        total_loss_value = 0
        step = 0
        for batch in dataset:
            step += 1
            x_batch, y_batch = batch[:, :-1], batch[:, -1]
            model_output, loss_value = test_step(x_batch, y_batch)
            Y_true.append(y_batch)
            Y_pred.append(model_output)
            total_loss_value += loss_value
        total_loss_value /= step

        if show:
            print('info: %s' % mode)
            # 画图
            Y_true = tf.concat(Y_true, axis=0).numpy()
            Y_pred = tf.squeeze(tf.concat(Y_pred, axis=0)).numpy()
            plt.figure(figsize=(20, 3))
            plt.plot(range(len(Y_pred)), Y_true, 'r', label="true")
            plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
            plt.legend(loc="upper right")  # 显示图中的标签
            plt.xlabel("ability")
            plt.ylabel('value of ablity')
            plt.show()

        return loss_value


    # ========================================== input layer ==========================================
    input_dict = dict()
    input_dict['input'] = tf.keras.Input(shape=(X.shape[1],), dtype=tf.int64, name='input')
    # ========================================== input layer ==========================================

    # ========================================== index layer ==========================================
    idx_output_dict = {}
    # normal feature idx
    for k in range(X.shape[1]):
        feature_input = tf.reshape(input_dict['input'][:, k], (-1, 1))
        feature_name = 'feature_%d' % k
        feature_vocabulary_list = np.arange(np.min(X), np.max(X))
        feature_num_oov_bucket = 1
        idx_layer = LookupLayer(vocabulary_list=feature_vocabulary_list,
                                vocabulary_type=tf.int64,
                                num_oov_buckets=feature_num_oov_bucket,
                                name=feature_name + '_idx_layer')
        embedding_num = len(feature_vocabulary_list) + feature_num_oov_bucket
        # Idx_output
        idx_output_dict[feature_name] = dict()
        idx_output = idx_layer(feature_input)
        idx_output_dict[feature_name]['idx_output'] = idx_output
        idx_output_dict[feature_name]['embedding_num'] = embedding_num
    # ========================================== index layer ==========================================

    # ======================================== embedding layer ========================================
    sfs, dfs = [], []
    # normal feature embedding
    for k in idx_output_dict.keys():
        idx_output = idx_output_dict[k]['idx_output']
        embedding_num = idx_output_dict[k]['embedding_num']
        feature_dense_dim = 1
        feature_sparse_dim = 1
        # sparse_layer
        sf_layer = Embedding(input_dim=embedding_num,
                             output_dim=feature_sparse_dim,
                             name=k + '_sparse_emb_layer')
        # dense_layer
        df_layer = Embedding(input_dim=embedding_num,
                             output_dim=feature_dense_dim,
                             name=k + '_dense_emb_layer')
        # sparse feature
        sf_output = sf_layer(idx_output)
        sf_output = tf.reshape(sf_output, (-1, sf_output.shape[-1]))
        # dense feature
        df_output = df_layer(idx_output)
        df_output = tf.reshape(df_output, (-1, df_output.shape[-1]))
        # Embedding Collect
        sfs.append(sf_output)
        dfs.append(df_output)

    # embedding
    feature_num = len(sfs)
    sf_embedding = tf.keras.layers.concatenate(sfs, axis=1)
    df_embedding = tf.keras.layers.concatenate(dfs, axis=1)
    df_embedding_split_by_field = tf.reshape(df_embedding, (-1, len(dfs), feature_dense_dim))
    # ======================================== embedding layer =======================================

    # ========================================== input layer ==========================================
    input_dict = dict()
    input_dict['input'] = tf.keras.Input(shape=(X.shape[1],), dtype=tf.int64, name='input')
    # ========================================== input layer ==========================================

    # ========================================== index layer ==========================================
    idx_output_dict = {}
    # normal feature idx
    for k in range(X.shape[1]):
        feature_input = tf.reshape(input_dict['input'][:, k], (-1, 1))
        feature_name = 'feature_%d' % k
        feature_vocabulary_list = np.arange(np.min(X), np.max(X))
        feature_num_oov_bucket = 1
        idx_layer = LookupLayer(vocabulary_list=feature_vocabulary_list,
                                vocabulary_type=tf.int64,
                                num_oov_buckets=feature_num_oov_bucket,
                                name=feature_name + '_idx_layer')
        embedding_num = len(feature_vocabulary_list) + feature_num_oov_bucket
        # Idx_output
        idx_output_dict[feature_name] = dict()
        idx_output = idx_layer(feature_input)
        idx_output_dict[feature_name]['idx_output'] = idx_output
        idx_output_dict[feature_name]['embedding_num'] = embedding_num
    # ========================================== index layer ==========================================

    # ======================================== embedding layer ========================================
    sfs, dfs = [], []
    # normal feature embedding
    for k in idx_output_dict.keys():
        idx_output = idx_output_dict[k]['idx_output']
        embedding_num = idx_output_dict[k]['embedding_num']
        feature_dense_dim = 1
        feature_sparse_dim = 1
        # sparse_layer
        sf_layer = Embedding(input_dim=embedding_num,
                             output_dim=feature_sparse_dim,
                             name=k + '_sparse_emb_layer')
        # dense_layer
        df_layer = Embedding(input_dim=embedding_num,
                             output_dim=feature_dense_dim,
                             name=k + '_dense_emb_layer')
        # sparse feature
        sf_output = sf_layer(idx_output)
        sf_output = tf.reshape(sf_output, (-1, sf_output.shape[-1]))
        # dense feature
        df_output = df_layer(idx_output)
        df_output = tf.reshape(df_output, (-1, df_output.shape[-1]))
        # Embedding Collect
        sfs.append(sf_output)
        dfs.append(df_output)

    # embedding
    feature_num = len(sfs)
    sf_embedding = tf.keras.layers.concatenate(sfs, axis=1)
    df_embedding = tf.keras.layers.concatenate(dfs, axis=1)
    df_embedding_split_by_field = tf.reshape(df_embedding, (-1, len(dfs), feature_dense_dim))
    # ======================================== embedding layer ========================================

    ################################################ Dataset ################################################
    global global_batch_size
    global_batch_size = 8
    trainset = tf.data.Dataset.from_tensor_slices(
        np.column_stack((X_train.astype(np.float32), Y_train.astype(np.float32))))
    trainset = trainset.shuffle(buffer_size=X.shape[0]).batch(global_batch_size)

    testset = tf.data.Dataset.from_tensor_slices(
        np.column_stack((X_test.astype(np.float32), Y_test.astype(np.float32))))
    testset = testset.batch(global_batch_size)

    ################################################ Model ################################################
    # IPNN
    ip_layer = IPLayer()
    ip_embedding = tf.concat([df_embedding, ip_layer(df_embedding_split_by_field)], axis=1)
    model = DNNLayer(unit_list=ipnn_shape,
                     activation_list=['relu'] * (len(ipnn_shape) - 1) + [None],
                     bias_list=[True] * len(ipnn_shape),
                     kernel_regularizer_list=[None] * len(ipnn_shape))
    output = model(ip_embedding)

    model = tf.keras.Model(input_dict, output)
    print('模型参数量: %d' % model.count_params())

    ################################################ Run ################################################
    train_loss, test_loss = [], []
    optimizer = keras.optimizers.Adam(1e-2)
    MSE_loss_fn = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    ################################################ Loop ################################################
    train_loss, test_loss = [], []
    train_loss.append(info('pretrained_trainset', model, trainset, False))
    test_loss.append(info('pretrained_testset', model, testset, False))

    epochs = 30
    for epoch in range(1, epochs + 1):
        print(f"--------  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Start of epoch {epoch}/{epochs}  --------",
              end='\r')
        step = 0
        total_loss_value = 0
        for train_batch in trainset:
            step += 1
            x_batch, y_batch = train_batch[:, :-1], train_batch[:, -1]
            _, loss_value = train_step(x_batch, y_batch)
            total_loss_value += loss_value
            # 保留数据
            train_loss.append(info('trainset_epo%d' % epoch, model, trainset, False))
            test_loss.append(info('testset_epo%d' % epoch, model, testset, False))

    # 画图
    train_loss = tf.concat(train_loss, axis=0).numpy()
    test_loss = tf.concat(test_loss, axis=0).numpy()
    # plt.figure(figsize=(8,6))
    # plt.plot(range(len(train_loss)), train_loss, 'r', label="train")
    # plt.plot(range(len(test_loss)), test_loss, 'b', label="test")
    # plt.legend(loc="upper right")  # 显示图中的标签
    # plt.xlabel("epoch")
    # plt.ylabel('loss value')
    # plt.ylim(0,0.2)
    # plt.show()

    result_loss.append(test_loss)