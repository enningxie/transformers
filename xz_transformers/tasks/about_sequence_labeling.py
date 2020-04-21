# -*- coding: utf-8 -*-
# Created by xieenning at 2020/4/16
"""Sequence labeling"""
import os
import time
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from xz_transformers.file_utils import ROOT_PATH, CONFIG_NAME, ID2LABEL_NAME
from xz_transformers.configuration_bert import BertConfig
from xz_transformers.tokenization_bert import load_vocab, PRETRAINED_VOCAB_FILES_MAP, BertTokenizer
from xz_transformers.modeling_tf_bert import TFBertForTokenClassification, TFBertForTokenClassificationWithCRF
from xz_transformers.data_processors import SequenceLabelingProcessor, convert_examples_to_features_labeling
from xz_transformers.modeling_tf_utils import calculate_steps, load_serialized_data
from xz_transformers.tokenizer import Tokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(level=logging.ERROR, format='%(message)s')


class SequenceLabeling(object):
    """
    序列标注任务
    """

    def __init__(self, pretrained_model_name, label_path, max_length, saved_model_path, loss_type='crossentropy'):
        """
        :param pretrained_model_name: 预训练模型权重保存简称或路径
        :param num_labels: 类别个数，1默认最后使用sigmoid，2则使用softmax
        :param max_length: 训练阶段sequence最大长度
        """
        self.data_processor = SequenceLabelingProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False,
                                                       do_basic_tokenize=True)
        if os.path.isfile(pretrained_model_name):
            self.tokenizer_ = Tokenizer(load_vocab(pretrained_model_name))

            self.pretrained_model_name = pretrained_model_name.split('/')[-2]
        else:
            self.tokenizer_ = Tokenizer(load_vocab(PRETRAINED_VOCAB_FILES_MAP['vocab_file'][pretrained_model_name]))
            # training parameters
            self.pretrained_model_name = pretrained_model_name
        self.labels = self.data_processor.get_labels(label_path)
        self.num_labels = len(self.labels)
        self.max_length = max_length

        self.saved_model_path = saved_model_path

        self.task = 'sl'

        self.loss_type = loss_type

    def generate_tf_dataset(self, data_path, batch_size):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path: 数据保存路径
        :param batch_size: 训练阶段 batch_size大小
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)[:1000]
        valid_examples = self.data_processor.get_dev_examples(data_path)

        # calculate steps
        train_steps = calculate_steps(len(train_examples), batch_size)
        valid_steps = calculate_steps(len(valid_examples), batch_size)

        # convert tasks to tf_dataset
        train_dataset = convert_examples_to_features_labeling(train_examples,
                                                              self.labels,
                                                              self.max_length,
                                                              self.tokenizer,
                                                              return_tensors='tf',
                                                              save_id2label_path=os.path.join(self.saved_model_path,
                                                                                              'id2label.pkl'))
        valid_dataset = convert_examples_to_features_labeling(valid_examples,
                                                              self.labels,
                                                              self.max_length,
                                                              self.tokenizer,
                                                              return_tensors='tf')

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
        if self.loss_type == 'crf':
            trained_model = TFBertForTokenClassificationWithCRF.from_pretrained(self.saved_model_path,
                                                                                config=trained_config)
        else:
            trained_model = TFBertForTokenClassification.from_pretrained(self.saved_model_path, config=trained_config)
        return trained_model

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
        if self.loss_type == 'crf':
            model = TFBertForTokenClassificationWithCRF.from_pretrained(self.pretrained_model_name, config=config)
            train_metric = tf.keras.metrics.Accuracy(name='accuracy')
            val_metric = tf.keras.metrics.Accuracy(name='accuracy')
        else:
            model = TFBertForTokenClassification.from_pretrained(self.pretrained_model_name, config=config)
            train_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            val_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)
        # Iterate over the batches of the dataset.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        # Iterate over epochs.
        best_val_acc = 0.
        for epoch in range(epochs):
            print('*********************')
            print('Epoch {} training...'.format(epoch))
            training_bar = tf.keras.utils.Progbar(train_steps, stateful_metrics=['loss', 'acc'])
            # Iterate over the batches of the dataset.
            for train_step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    sequence_lengths = tf.math.reduce_sum(
                        tf.cast(tf.math.not_equal(x_batch_train['input_ids'], 0), dtype=tf.int32), axis=-1)
                    if self.loss_type == 'crossentropy':
                        logits = model(x_batch_train, training=True)[0]  # Logits for this minibatch
                        masks = tf.sequence_mask(
                            sequence_lengths, maxlen=tf.shape(logits)[1], dtype=tf.bool)
                        active_logits = tf.boolean_mask(logits, masks)
                        active_labels = tf.boolean_mask(y_batch_train, masks)
                        cross_entropy = loss_fn(active_labels, active_logits)
                        loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)

                    else:
                        tmp_result = model(x_batch_train, training=True)  # Logits for this minibatch
                        logits, logits_ = tmp_result[0], tmp_result[1]
                        masks = tf.sequence_mask(
                            sequence_lengths, maxlen=tf.shape(logits_)[1], dtype=tf.bool)

                        log_likelihood, _ = tfa.text.crf_log_likelihood(logits_, y_batch_train,
                                                                        sequence_lengths,
                                                                        model.crf.trans_params)
                        loss = - tf.reduce_mean(log_likelihood)
                        active_logits = tf.boolean_mask(logits, masks)
                        active_labels = tf.boolean_mask(y_batch_train, masks)
                    # Update training metric.
                    train_metric(active_labels, active_logits)
                    tmp_accuracy = train_metric.result()
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # Logging
                training_bar.update(train_step + 1, values=[('loss', float(loss)), ('acc', float(tmp_accuracy))])

            # Reset training metrics at the end of each epoch
            train_metric.reset_states()
            validating_bar = tf.keras.utils.Progbar(valid_steps, stateful_metrics=['val_acc'])
            # Run a validation loop at the end of each epoch.
            for val_step, (x_batch_val, y_batch_val) in enumerate(valid_dataset):
                sequence_lengths_val = tf.math.reduce_sum(
                    tf.cast(tf.math.not_equal(x_batch_val['input_ids'], 0), dtype=tf.int32), axis=-1)
                if self.loss_type == 'crf':
                    val_logits, val_logits_ = model(x_batch_val, training=False)
                    masks = tf.sequence_mask(
                        sequence_lengths_val, maxlen=tf.shape(val_logits_)[1], dtype=tf.bool)
                else:
                    val_logits = model(x_batch_val, training=False)[0]
                    masks = tf.sequence_mask(
                        sequence_lengths_val, maxlen=tf.shape(val_logits)[1], dtype=tf.bool)
                active_logits_val = tf.boolean_mask(val_logits, masks)
                active_labels_val = tf.boolean_mask(y_batch_val, masks)
                # Update val metrics
                val_metric(active_labels_val, active_logits_val)
                tmp_accuracy = val_metric.result()
                # Logging
                validating_bar.update(val_step + 1, values=[('val_acc', float(tmp_accuracy))])
            val_acc = val_metric.result()
            # Save the best model with the highest verification accuracy
            if val_acc > best_val_acc:
                print('model saving...')
                # normal
                model.save_pretrained(self.saved_model_path)
                best_val_acc = val_acc
            val_metric.reset_states()
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
        if self.loss_type == 'crf':
            evaluate_metric = tf.keras.metrics.Accuracy(name='accuracy')
        else:
            evaluate_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # 原始数据转tf-dataset
        test_examples = self.data_processor.get_test_examples(test_data_path)
        test_steps = calculate_steps(len(test_examples), batch_size)
        test_dataset = convert_examples_to_features_labeling(test_examples,
                                                             self.labels,
                                                             self.max_length,
                                                             self.tokenizer,
                                                             return_tensors='tf')
        test_dataset = test_dataset.batch(batch_size)
        # model evaluation step
        evaluation_start_time = time.time()
        test_bar = tf.keras.utils.Progbar(test_steps, stateful_metrics=['test_loss', 'test_acc'])
        # Run a validation loop at the end of each epoch.
        for test_step, (x_batch_test, y_batch_test) in enumerate(test_dataset):
            sequence_lengths_test = tf.math.reduce_sum(
                tf.cast(tf.math.not_equal(x_batch_test['input_ids'], 0), dtype=tf.int32), axis=-1)
            if self.loss_type == 'crf':
                test_logits, test_logits_ = trained_model(x_batch_test, training=False)
                masks = tf.sequence_mask(
                    sequence_lengths_test, maxlen=tf.shape(test_logits_)[1], dtype=tf.bool)
                log_likelihood, _ = tfa.text.crf_log_likelihood(test_logits_, y_batch_test,
                                                                sequence_lengths_test,
                                                                trained_model.crf.trans_params)
                loss = - tf.reduce_mean(log_likelihood)
                active_logits_test = tf.boolean_mask(test_logits, masks)
                active_labels_test = tf.boolean_mask(y_batch_test, masks)
            else:
                test_logits = trained_model(x_batch_test, training=False)[0]
                masks = tf.sequence_mask(
                    sequence_lengths_test, maxlen=tf.shape(test_logits)[1], dtype=tf.bool)
                active_logits_test = tf.boolean_mask(test_logits, masks)
                active_labels_test = tf.boolean_mask(y_batch_test, masks)
                cross_entropy = loss_fn(active_labels_test, active_logits_test)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)
            # Update val metrics
            evaluate_metric(active_labels_test, active_logits_test)
            tmp_accuracy = evaluate_metric.result()
            # Logging
            test_bar.update(test_step + 1, values=[('test_acc', float(loss)), ('val_acc', float(tmp_accuracy))])
        cost_time = time.time() - evaluation_start_time
        with open(os.path.join(self.saved_model_path, 'evaluate.log'), 'w') as f:
            f.write(f'Evaluation cost time: {cost_time}\nEvaluate loss: {loss}\n'
                    f'Evaluate accuracy: {tmp_accuracy}')

    def predict_op(self, trained_model, batch_text):
        """
        模型预测阶段
        :param trained_model: 训练好了的模型
        :param batch_text_pairs: [['想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'], ['今天天气很差', '今天天气很棒']]
        :return:
        """
        inputs = self.tokenizer.batch_encode_plus(batch_text, max_length=self.max_length, return_tensors="tf",
                                                  pad_to_max_length=True)
        tmp_id2label_path = os.path.join(self.saved_model_path, ID2LABEL_NAME)
        id2label = load_serialized_data(tmp_id2label_path)
        if self.loss_type == 'crf':
            logits, logits_ = trained_model(inputs, training=False)
            tmp_pred_label = list(logits.numpy())
            print('d')
        else:
            logits = trained_model(inputs, training=False)[0]
            tmp_pred = tf.nn.softmax(logits, axis=-1)
            tmp_pred_label = list(tf.argmax(tmp_pred, axis=-1).numpy())
            print('d')
        tmp_pred_label_ = list(map(list, tmp_pred_label))
        return [list(map(id2label.get, pred_label)) for pred_label in tmp_pred_label_]


if __name__ == '__main__':
    # raw_data_path = os.path.join(ROOT_PATH, 'data/ner')
    raw_data_path = '/Data/public/DataSets/ner'
    tmp_saved_model_path = os.path.join(ROOT_PATH, 'saved_models/ner/chinese-rbt3-03')
    tmp_sl_obj = SequenceLabeling('chinese-rbt3', os.path.join(raw_data_path, 'labels.txt'), 128,
                                  saved_model_path=tmp_saved_model_path, loss_type='crf')
    # #### training step ####
    # tmp_sl_obj.train_op(raw_data_path, epochs=2, batch_size=16)

    # #### evaluating step ####
    # tmp_sl_obj.evaluate_op(raw_data_path, batch_size=16)

    # #### predicting step ####
    # trained_model = tmp_sl_obj.get_trained_model()
    # batch_text = ['这次投票结果，预计在星期一公布。']
    # tmp_result = tmp_sl_obj.predict_op(trained_model, batch_text)
    # print(tmp_result)
