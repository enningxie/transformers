# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
"""使用LCQMC数据集训练语义匹配模型"""
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from xz_transformers.file_utils import ROOT_PATH, CONFIG_NAME
from xz_transformers.configuration_bert import BertConfig
from xz_transformers.tokenization_bert import BertTokenizer, load_vocab, PRETRAINED_VOCAB_FILES_MAP
from xz_transformers.modeling_tf_bert import TFBertForSequenceClassification
from xz_transformers.data_processors import SequencePairClassificationProcessor, convert_examples_to_features
from xz_transformers.modeling_tf_utils import calculate_steps
from xz_transformers.tokenizer import Tokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class SequencePairClassification:
    """
    句子对分类，相似不相似
    """

    def __init__(self, pretrained_model_name, num_labels, max_length, saved_model_path):
        """
        :param pretrained_model_name: 预训练模型权重保存简称或路径
        :param num_labels: 类别个数，1默认最后使用sigmoid，2则使用softmax
        :param max_length: 训练阶段sequence最大长度
        """
        self.data_processor = SequencePairClassificationProcessor()
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False,
                                                       do_basic_tokenize=True)
        if os.path.isfile(pretrained_model_name):
            self.tokenizer_ = Tokenizer(load_vocab(pretrained_model_name))

            self.pretrained_model_name = pretrained_model_name.split('/')[-2]
        else:
            self.tokenizer_ = Tokenizer(load_vocab(PRETRAINED_VOCAB_FILES_MAP['vocab_file'][pretrained_model_name]))
            # training parameters
            self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.max_length = max_length

        self.saved_model_path = saved_model_path

        if num_labels == 1:
            self.mode = 'binary'
        else:
            self.mode = 'category'

        self.task = 'spc'

    def generate_tf_dataset(self, data_path, batch_size):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path: 数据保存路径
        :param batch_size: 训练阶段 batch_size大小
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)
        valid_examples = self.data_processor.get_dev_examples(data_path)

        # calculate steps
        train_steps = calculate_steps(len(train_examples), batch_size)
        valid_steps = calculate_steps(len(valid_examples), batch_size)

        # convert tasks to tf_dataset
        train_dataset = convert_examples_to_features(train_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task, return_tensors='tf')
        valid_dataset = convert_examples_to_features(valid_examples, self.tokenizer, max_length=self.max_length,
                                                     task=self.task, return_tensors='tf')

        # preprocess tf_dataset
        train_dataset = train_dataset.shuffle(10000).batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)

        return (train_dataset, train_steps), (valid_dataset, valid_steps)

    def get_trained_model(self):
        """
        加载训练好的模型
        :return:
        """
        trained_config = BertConfig.from_pretrained(os.path.join(self.saved_model_path, CONFIG_NAME))
        trained_model = TFBertForSequenceClassification.from_pretrained(self.saved_model_path, config=trained_config)
        return trained_model

    def get_compiled_model(self, model):
        """
        返回编译后的模型
        :param model: 编译前的模型
        :return:
        """
        # prepare training: compile tf.keras model with optimizer, loss and learning rate schedule
        if self.mode == 'binary':
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metric = tf.keras.metrics.BinaryAccuracy('accuracy')
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()
        return model

    def train_op(self, data_path, epochs, batch_size):
        """
        模型训练
        :param data_path: 原始数据保存路径
        :param epochs: 训练阶段迭代次数
        :param batch_size: 训练阶段batch_size
        :param saved_model_path: 训练结束后模型保存路径
        :return:
        """
        config = BertConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        model = TFBertForSequenceClassification.from_pretrained(self.pretrained_model_name, config=config)
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)
        model = self.get_compiled_model(model)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        # Train and evaluate using tf.keras.Model.fit()
        training_start_time = time.time()
        history_train = model.fit(train_dataset.repeat(), epochs=epochs, steps_per_epoch=train_steps,
                                  validation_data=valid_dataset.repeat(), validation_steps=valid_steps,
                                  callbacks=[early_stop])
        cost_time = time.time() - training_start_time
        model.save_pretrained(self.saved_model_path)
        with open(os.path.join(self.saved_model_path, 'training.log'), 'w') as f:
            f.write(f'Training cost time: {cost_time}\nTraining history: {history_train.history}')
        return model

    def evaluate_op(self, test_data_path, batch_size=64):
        """
        模型evaluation step
        :param test_data_path: 测试数据保存路径
        :param batch_size: evaluation阶段的batch_size
        :return:
        """
        # 模型加载
        trained_model = self.get_trained_model()
        trained_model = self.get_compiled_model(trained_model)
        # 原始数据转tf-dataset
        test_examples = self.data_processor.get_test_examples(test_data_path)
        test_steps = calculate_steps(len(test_examples), batch_size)
        test_dataset = convert_examples_to_features(test_examples, self.tokenizer, max_length=self.max_length,
                                                    task=self.task, return_tensors='tf')
        test_dataset = test_dataset.batch(batch_size)
        # model evaluation step
        evaluation_start_time = time.time()
        evaluation_loss, evaluation_acc = trained_model.evaluate(test_dataset, steps=test_steps)
        cost_time = time.time() - evaluation_start_time
        print(f"Evaluate step loss: {evaluation_loss}")
        print(f"Evaluate step accuracy: {evaluation_acc}")
        with open(os.path.join(self.saved_model_path, 'evaluate.log'), 'w') as f:
            f.write(f'Evaluation cost time: {cost_time}\nEvaluate loss: {evaluation_loss}\n'
                    f'Evaluate accuracy: {evaluation_acc}')

    def predict_op(self, trained_model, batch_text_pairs):
        """
        模型预测阶段
        :param trained_model: 训练好了的模型
        :param batch_text_pairs: [['想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'], ['今天天气很差', '今天天气很棒']]
        :return:
        """
        batch_text_pairs_ = []
        for sentence1, sentence2 in batch_text_pairs:
            batch_text_pairs_.append(self.tokenizer_.tokenize_once(sentence1, sentence2))
        inputs = self.tokenizer.batch_encode_plus(batch_text_pairs_, max_length=self.max_length,
                                                  return_tensors='tf', pad_to_max_length=True)
        # inputs_ = [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['position_ids']]
        tmp_start_time = time.time()
        # tmp_pred = trained_model(inputs)
        tmp_pred = trained_model.predict(inputs, batch_size=1024)
        tmp_cost_time = time.time() - tmp_start_time
        if self.mode == 'binary':
            tmp_result = tf.nn.sigmoid(tmp_pred)
            return np.squeeze(tmp_result.numpy(), axis=-1), tmp_cost_time
        else:
            tmp_result = tf.nn.softmax(tmp_pred, axis=-1)
            return tmp_result.numpy(), tmp_cost_time


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/LCQMC')
    # tmp_saved_model_path = '/Data/enningxie/Codes/transformers_xz/saved_models/intent_detection_0515/chinese-rbt3'
    # tmp_saved_model_path = '/Data/enningxie/Pretrained_models/chinese-rbt3'
    tmp_saved_model_path = '/Data/xen/Codes/xz_transformers/saved_models/intent_detection_0511/chinese-roberta-wwm-ext/'
    # tmp_saved_model_path = '/Data/enningxie/Codes/transformers_xz/saved_models/intent_detection_2_10_0_onnx/chinese-rbt3'
    tmp_spc_obj = SequencePairClassification('chinese-roberta-wwm-ext',
                                             num_labels=1,
                                             max_length=64,
                                             saved_model_path=tmp_saved_model_path)
    # #### training step ####
    # tmp_spc_obj.train_op(raw_data_path, epochs=15, batch_size=64)

    # tmp_trained_model_path = os.path.join(ROOT_PATH, 'tasks/saved_models/spc_1')

    # #### evaluate step ####
    # tmp_spc_obj.evaluate_op(raw_data_path)

    # #### predict step ####
    # # process batch_text_pairs
    # # data/LCQMC/test.tsv
    # # data/sequence_pair/custom_df_01.tsv
    # valuable_data_path = '/Data/xen/Codes/notebooks/Intent_detection/data/test_df.tsv'
    # valuable_data_path_ = '/Data/xen/Codes/xz_transformers/data/LCQMC/test.tsv'
    # valuable_df = pd.read_csv(valuable_data_path, sep='\t')
    # tmp_batch_text_pairs = []
    # tmp_label = []
    # for _, tmp_row in valuable_df.iterrows():
    #     tmp_batch_text_pairs.append((tmp_row.sentence1, tmp_row.sentence2))
    #     tmp_label.append(tmp_row.label)
    # trained_model = tmp_spc_obj.get_trained_model()
    # tmp_start_time = time.perf_counter()
    # tmp_pred = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
    # print(f'Total cost time: {time.perf_counter() - tmp_start_time}')
    # tmp_threshold = 0.01
    # best_threshold = tmp_threshold
    # best_precision = 0.0
    # best_recall = 0.0
    # best_f1_score = 0.0
    # best_accuracy = 0.0
    # while tmp_threshold < 1.0:
    #     tmp_y_true = np.asarray(tmp_label)
    #     tmp_y_pred = (tmp_pred > tmp_threshold).astype(np.int64)
    #     tn, fp, fn, tp = confusion_matrix(tmp_y_true, tmp_y_pred).ravel()
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     f1_score = 2 * (precision * recall) / (precision + recall)
    #     accuracy = (tp + tn) / (tn + fp + fn + tp)
    #     if f1_score > best_f1_score:
    #         best_threshold = tmp_threshold
    #         best_precision = precision
    #         best_recall = recall
    #         best_f1_score = f1_score
    #         best_accuracy = accuracy
    #     tmp_threshold += 0.01
    # print(f"--> threshold: {best_threshold}.")
    # print(f"--> precision: {best_precision}.")
    # print(f"--> recall: {best_recall}.")
    # print(f"--> f1 score: {best_f1_score}.")
    # print(f"--> accuracy score: {best_accuracy}.")

    # #### play with model ####
    # trained_model = tmp_spc_obj.get_trained_model()
    # while True:
    #     sentence1 = input('sent1: ')
    #     sentence2 = input('sent2: ')
    #     tmp_batch_text_pairs = [(sentence1, sentence2)]
    #     tmp_start_time = time.time()
    #     tmp_pred = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
    #     print(f'cost time {time.time() - tmp_start_time}')
    #     print(f'pred: {tmp_pred[0]}')
    #     tmp_y_pred = (tmp_pred > 0.2).astype(np.int64)
    #     print(f'y_pred: {tmp_y_pred[0]}\n++++++++++++++++++++++++++++++++')

    # # Test performance for benchmarking.
    # tmp_batch_text_pairs = [("一二三四五一二三四五一二三四五一二三四五一二三四五一二三四五一二三四五", "一二三四五六一二三四五六一二三四五六一二三四五六一二三四五六一二三四五六")] + \
    #                        [("一二三四五", "一二三四五")] * 99
    # trained_model = tmp_spc_obj.get_trained_model()
    # tmp_start_time = time.perf_counter()
    # tmp_pred = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
    # print(f'Total cost time: {time.perf_counter() - tmp_start_time}')

    #### predict step ####
    tmp_batch_text_pairs = [('！？解下', '阿斯顿'), ('！？解asdsa下', '阿斯sadfa顿'), ('！？解fasfsdfsdf下', '阿sasdsfsfsdfs斯顿')]
    trained_model = tmp_spc_obj.get_trained_model()
    print(trained_model.name)
    tmp_start_time = time.time()
    tmp_pred, tmp_cost_time = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
    print(f'Total cost time: {time.time() - tmp_start_time}')
    print(f'Inference cost time: {tmp_cost_time}')
    print(f'++++++++++++++++++++++++++++++++++++++++++++++++')
    tmp_start_time = time.time()
    tmp_batch_text_pairs = [('想了解下您会想看哪款车型想了解下您会想看哪款车型想了解下您会想看', '是想请问下您当时买的是哪款车呢想了解下您会想看哪款车型想了解下您')] * 100
    tmp_pred, tmp_cost_time = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
    print(f'Total cost time: {time.time() - tmp_start_time}')
    print(f'Inference cost time: {tmp_cost_time}')
    # for tmp_pair, tmp_score in zip(tmp_batch_text_pairs, tmp_pred):
    #     print(f'{tmp_pair[0]} & {tmp_pair[1]} --> {tmp_score}')
    print(tmp_pred)

    warm_up_turns = 10
    total_turns = 100

    for _ in range(warm_up_turns):
        tmp_pred, _ = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)

    total_time_origin = []
    total_time_origin_ = []
    for _ in range(total_turns):
        tmp_start_time_origin = time.perf_counter()
        tmp_pred, tmp_cost_time_origin = tmp_spc_obj.predict_op(trained_model, tmp_batch_text_pairs)
        total_time_origin.append(time.perf_counter() - tmp_start_time_origin)
        total_time_origin_.append(tmp_cost_time_origin)

    total_time_origin_sorted = sorted(total_time_origin)
    total_time_origin_sorted_ = sorted(total_time_origin_)

    print(f'Origin model total cost time: {total_time_origin_sorted[-2]}')
    print(f'Origin model tmp cost time: {total_time_origin_sorted_[-2]}')