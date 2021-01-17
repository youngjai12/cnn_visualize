import tensorflow
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Input, Reshape
from tensorflow.keras.regularizers import l2
import yaml

def loss_wrapper(weights_loss):
    def custom_loss(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits()
def get_yamldict(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.Loader)
    return yaml_dict
def getModel(word_seq_len, char_seq_len, ngram_seq_len, chars_dict_len, word_dict_len, ngram_dict_len, reg_lambda,
             emb_dim):
    ########### char-level embedding, conv layer #############
    char_input = Input(shape=(char_seq_len), name="char_input")
    print("shape of char_input: ", char_input.shape)
    emb_char = Embedding(chars_dict_len, emb_dim)(char_input)
    print("char-Embedding: ", emb_char.shape)
    pooled_result = []
    for filter_size in [3, 4, 5, 6]:
        char_conv_output = Conv1D(filters=256, kernel_size=filter_size, strides=1, activation="relu", padding="VALID")(
            emb_char)
        print("char-convOutput : ", char_conv_output.shape)
        max_pooled_output = MaxPooling1D(pool_size=char_seq_len  - filter_size + 1, strides=1, padding="VALID")(
            char_conv_output)
        print("char - maxPool shape: ", max_pooled_output.shape)
        pooled_result.append(max_pooled_output)
    num_filters_total = 256 * 4
    filter_pooled = tensorflow.concat(pooled_result, axis=1)
    print("4 filter output pooled: ", filter_pooled.shape)
    char_flat = tensorflow.reshape(filter_pooled, [-1, num_filters_total])
    char_drop_out = Dropout(0.5)(char_flat)
    ######### word-level embedding, conv_layer ###########
    input_word = Input(shape=(word_seq_len), name="word_input")
    input_ngram = Input(shape=(word_seq_len * ngram_seq_len), name="ngram_input")
    reshaping_layer = Reshape((word_seq_len, ngram_seq_len), input_shape=(word_seq_len * ngram_seq_len,))
    reshepd_ngram = reshaping_layer(input_ngram)
    # input_ngram_padded = Input(shape=(200,20,32,), name="ngram_padded_input")
    emb_word = Embedding(word_dict_len + 1, emb_dim)(input_word)
    print("word-embedding: ", emb_word.shape)
    emb_ngram = Embedding(ngram_dict_len + 1, emb_dim)(reshepd_ngram)
    emb_ngram_sum = tensorflow.reduce_sum(emb_ngram, 2)
    tot_word_emb = tensorflow.add(emb_ngram_sum, emb_word)
    word_pooled_result = []
    for filter_size in [3, 4, 5, 6]:
        layer_name = "word_layer_fsize_" + str(filter_size)
        word_conv_output = Conv1D(filters=256, kernel_size=filter_size, strides=1, activation="relu", padding="VALID",
                                  name=layer_name)(tot_word_emb)
        max_pooled_output_word = MaxPooling1D(pool_size=word_seq_len - filter_size + 1, strides=1, padding="VALID")(
            word_conv_output)
        word_pooled_result.append(max_pooled_output_word)
    word_filter_pooled = tensorflow.concat(word_pooled_result, axis=1)
    word_flat = tensorflow.reshape(word_filter_pooled, [-1, num_filters_total])
    print("word_flat shape: ", word_flat.shape)
    word_drop_out = Dropout(0.5)(word_flat)
    ############# Fully connected Layer ##########
    char_dense_layer = Dense(512, input_shape=(num_filters_total,), kernel_initializer="glorot_uniform",
                             kernel_regularizer=l2(reg_lambda))(char_drop_out)
    word_dense_layer = Dense(512, input_shape=(num_filters_total,), kernel_initializer="glorot_uniform",
                             kernel_regularizer=l2(reg_lambda))(word_drop_out)
    conv_final_output = tensorflow.concat([char_dense_layer, word_dense_layer], 1)
    ############### Dense layer - before_softmax #######################
    d0 = Dense(512, input_shape=(1024,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(reg_lambda))(
        conv_final_output)
    d1 = Dense(256, input_shape=(512,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(reg_lambda))(d0)
    d2 = Dense(128, input_shape=(256,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(reg_lambda))(d1)
    score = Dense(2, input_shape=(128,), kernel_initializer="glorot_uniform", kernel_regularizer=l2(reg_lambda))(d2)
    ############### Score & Predictions & Probability ##############
    predictions = tensorflow.argmax(score, 1, name="predictions")
    prob = tensorflow.nn.softmax(score, name="prob")
    model = tensorflow.keras.Model([char_input, input_word, input_ngram], prob)
    return model

def input_fn(path, maxWordPerUrl, maxCharPerUrl, maxNgramPerUrl, shuffle_buffer_size, batch_size):
    dataset = tensorflow.data.TFRecordDataset(path)
    feature_map = {
        "engineeredChar": tensorflow.io.FixedLenFeature((maxCharPerUrl,), tensorflow.int64),
        "engineeredWord": tensorflow.io.FixedLenFeature((maxWordPerUrl,), tensorflow.int64),
        "engineeredNgram": tensorflow.io.FixedLenFeature((maxWordPerUrl * maxNgramPerUrl,), tensorflow.int64),
        "label": tensorflow.io.FixedLenFeature([], tensorflow.int64)
    }
    def _parse_fn(record, feature_map):
        example = tensorflow.io.parse_single_example(serialized=record, features=feature_map)
        onehot_label = tensorflow.cast(tensorflow.one_hot(tensorflow.where(tensorflow.equal(example["label"], 1), 1, 0), depth=2), dtype=tensorflow.int64)
        return {"char_input": example["engineeredChar"], "word_input": example["engineeredWord"],
                "ngram_input": example["engineeredNgram"]}, onehot_label
    dataset = dataset.map(lambda record: _parse_fn(record, feature_map))
    return dataset.shuffle(shuffle_buffer_size).batch(batch_size)
def read_tfrecord(path, maxWordPerUrl, maxCharPerUrl, maxNgramPerUrl, read_batch_size):
    dataset = tensorflow.data.TFRecordDataset(path)
    feature_map = {
        "engineeredChar": tensorflow.io.FixedLenFeature((maxCharPerUrl,), tensorflow.int64),
        "engineeredWord": tensorflow.io.FixedLenFeature((maxWordPerUrl,), tensorflow.int64),
        "engineeredNgram": tensorflow.io.FixedLenFeature((maxWordPerUrl * maxNgramPerUrl,), tensorflow.int64),
        "label": tensorflow.io.FixedLenFeature([], tensorflow.int64)
    }
    def _parse_fn(record, feature_map):
        example = tensorflow.io.parse_single_example(serialized=record, features=feature_map)
        onehot_label = tensorflow.cast(tensorflow.one_hot(tensorflow.where(tensorflow.equal(example["label"], 1), 1, 0), depth=2), dtype=tensorflow.int64)
        return {"char_input": example["engineeredChar"], "word_input": example["engineeredWord"],
                "ngram_input": example["engineeredNgram"]}, onehot_label
    dataset = dataset.map(lambda record: _parse_fn(record, feature_map))
    return dataset.batch(read_batch_size)
 
def input_small_fn(path, maxWordPerUrl, maxCharPerUrl, maxNgramPerUrl, num_urls):
    dataset = tensorflow.data.TFRecordDataset(path)
    feature_map = {
        "engineeredChar": tensorflow.io.FixedLenFeature((maxCharPerUrl,), tensorflow.int64),
        "engineeredWord": tensorflow.io.FixedLenFeature((maxWordPerUrl,), tensorflow.int64),
        "engineeredNgram": tensorflow.io.FixedLenFeature((maxWordPerUrl * maxNgramPerUrl,), tensorflow.int64),
        "label": tensorflow.io.FixedLenFeature([], tensorflow.int64)
    }
    def _parse_fn(record, feature_map):
        example = tensorflow.io.parse_single_example(serialized=record, features=feature_map)
        onehot_label = tensorflow.cast(tensorflow.one_hot(tensorflow.where(tensorflow.equal(example["label"], 1), 1, 0), depth=2), dtype=tensorflow.int64)
        return {"char_input": example["engineeredChar"], "word_input": example["engineeredWord"],
                "ngram_input": example["engineeredNgram"]}, onehot_label
    dataset = dataset.map(lambda record: _parse_fn(record, feature_map))
    return dataset.take(num_urls)
 
def get_label_and_urlstring(path):
    dataset = tensorflow.data.TFRecordDataset(path)
    feature_map = {
        "url": tensorflow.io.FixedLenFeature([], tensorflow.string),
        "label":tensorflow.io.FixedLenFeature([], tensorflow.int64),
    }
    def _parse_fn(record, feature_map):
        example = tensorflow.io.parse_single_example(serialized=record, features=feature_map)
        one_hot_label = tensorflow.cast(tensorflow.one_hot(tensorflow.where(tensorflow.equal(example["label"], 1),1,0), depth=2), dtype=tensorflow.int64)
        return {"url": example["url"], "label": one_hot_label}
    
    return dataset.map(lambda record: _parse_fn(record, feature_map))
    
    
def get_label_url_source(path):
    dataset = tensorflow.data.TFRecordDataset(path)
    feature_map = {
        "url": tensorflow.io.FixedLenFeature([], tensorflow.string),
        "label":tensorflow.io.FixedLenFeature([], tensorflow.int64),
        "urlSource":tensorflow.io.FixedLenFeature([], tensorflow.string)
    }
    def _parse_fn(record, feature_map):
        example = tensorflow.io.parse_single_example(serialized=record, features=feature_map)
        one_hot_label = tensorflow.cast(tensorflow.one_hot(tensorflow.where(tensorflow.equal(example["label"], 1),1,0), depth=2), dtype=tensorflow.int64)
        return {"url": example["url"], "label": one_hot_label, "urlSource": example["urlSource"]}
    
    dataset = dataset.map(lambda record: _parse_fn(record, feature_map))
    return dataset