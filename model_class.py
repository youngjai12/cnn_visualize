import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Dropout, Input, Reshape
from tensorflow.keras.regularizers import l2






class URLNet():
    CHAR_INPUT_LAYER_NAME = "char_input"
    WORD_INPUT_LAYER_NAME = "word_input"
    NGRAM_INPUT_LAYER_NAME = "ngram_input"

    def __init__(self, word_seq_len, char_seq_len, ngram_seq_len, chars_dict_len, word_dict_len, ngram_dict_len, reg_lambda, emb_dim, kernel_size=[3,4,5,6]):
        self.word_seq_len = word_seq_len
        self.char_seq_len = char_seq_len
        self.ngram_seq_len = ngram_seq_len
        self.chars_dict_len = chars_dict_len
        self.word_dict_len = word_dict_len
        self.ngram_dict_len = ngram_dict_len
        self.reg_lambda = reg_lambda
        self.emb_dim = emb_dim
        self.kernel_size_list = kernel_size


    def _cnn_layer(self, x, seq_len, kernels, feature_maps):
        pooled_outputs = []
        for kernel_idx, kernel_size in enumerate(kernels):
            conv_output = Conv1D(filters=256, kernel_size=kernel_size, strides=1,
                                      activation="relu", padding="VALID")(x)
            feature_maps.append(conv_output)
            pooled_output = MaxPool1D(pool_size=seq_len-kernel_size+1, strides=1, padding="VALID")(conv_output)
            pooled_outputs.append(pooled_output)
        kernel_pooled = tf.concat(pooled_outputs, axis=1)
        num_kernel_total = 256 * len(kernels)
        flattend = tf.reshape(kernel_pooled, [-1, num_kernel_total])
        return flattend

    def _score_layer(self,x, num_class, last_shape):
        return Dense(num_class, input_shape=(last_shape,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(self.reg_lambda))(x)

    def _fc_layer(self, x):
        d1 = Dense(512, input_shape=(1024,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(self.reg_lambda))(x)
        d2 = Dense(256, input_shape=(512,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(self.reg_lambda))(d1)
        d3 = Dense(128, input_shape=(256,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(self.reg_lambda))(d2)

        return d3, 128



    def build(self):
        char_input_layer = Input(shape=(self.char_seq_len), name=self.CHAR_INPUT_LAYER_NAME)
        word_input_layer = Input(shape=(self.word_seq_len), name = self.WORD_INPUT_LAYER_NAME)

        ngram_input_layer = Input(shape=(self.word_seq_len * self.ngram_seq_len), name = self.NGRAM_INPUT_LAYER_NAME)
        reshaping = Reshape((self.word_seq_len, self.ngram_seq_len), input_shape=(self.word_seq_len * self.ngram_seq_len,))
        reshaped_ngram_input = reshaping(ngram_input_layer)


        char_embedding = Embedding(self.chars_dict_len+1, self.emb_dim)(char_input_layer)
        word_embedding = Embedding(self.word_dict_len+1, self.emb_dim)(word_input_layer)
        ngram_embedding = Embedding(self.ngram_dict_len+1, self.emb_dim)(reshaped_ngram_input)
        ngram_sum_embed = tf.reduce_sum(ngram_embedding, axis=2)
        final_word_embed = tf.add(word_embedding, ngram_sum_embed)

        self.char_feature_maps=[]
        self.word_feature_maps=[]

        char_flatten_layer = self._char_cnn_layer(char_embedding, self.char_seq_len, self.kernel_size_list, self.char_feature_maps)
        word_flatten_layer = self._cnn_layer(final_word_embed, self.word_seq_len, self.kernel_size_list, self.word_feature_maps)

        char_drop_out = Dropout(0.5)(char_flatten_layer)
        word_drop_out = Dropout(0.5)(word_flatten_layer)
        num_kernel_total =  256 * len(self.kernel_size_list)

        word_layer = Dense(512, input_shape=(num_kernel_total,), kernel_initializer="glorot_uniform",
              kernel_regularizer=l2(self.reg_lambda))(word_drop_out)
        char_layer = Dense(512, input_shape=(num_kernel_total,), kernel_initializer="glorot_uniform",
              kernel_regularizer=l2(self.reg_lambda))(char_drop_out)

        conat_layer = tf.concat([word_layer, char_layer], axis=1)

        fc_layer, last_shape = self._fc_layer(conat_layer)

        self.score_layer = self._score_layer(fc_layer, 2, last_shape, name="prediction_score")
        self.prob = tf.nn.softmax(self.score_layer, name="prob")

        return tf.keras.Model([char_input_layer, word_input_layer, ngram_input_layer], self.prob)


model = URLNet()
model.build