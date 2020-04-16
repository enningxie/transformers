# -*- coding: utf-8 -*-
# Created by xieenning at 2020/4/16
"""Sequence labeling"""
import os
import time
import tensorflow as tf
import pandas as pd
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from xz_transformers.file_utils import ROOT_PATH, CONFIG_NAME
from xz_transformers.configuration_bert import BertConfig
from xz_transformers.tokenization_bert import load_vocab, PRETRAINED_VOCAB_FILES_MAP
from xz_transformers.modeling_tf_bert import TFBertForTokenClassification
from xz_transformers.data_processors import SequenceLabelingProcessor, convert_examples_to_features_labeling
from xz_transformers.modeling_tf_utils import calculate_steps
from xz_transformers.tokenizer import Tokenizer
from transformers import BertTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logging.basicConfig(level=logging.ERROR, format='%(message)s')


class SequenceLabeling(object):
    """
    序列标注任务
    """

    def __init__(self, pretrained_model_name, label_path, max_length, saved_model_path):
        """
        :param pretrained_model_name: 预训练模型权重保存简称或路径
        :param num_labels: 类别个数，1默认最后使用sigmoid，2则使用softmax
        :param max_length: 训练阶段sequence最大长度
        """
        self.data_processor = SequenceLabelingProcessor()
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # todo do_basic_tokenize set False for chinese
        self.tokenizer = BertTokenizer.from_pretrained(
            "/Data/xen/Codes/xz_transformers/pretrained_models/tensorflow2.x/chinese-rbt3", do_lower_case=False)
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
        model = TFBertForTokenClassification.from_pretrained(self.pretrained_model_name, config=config)
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)
        # Iterate over the batches of the dataset.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        val_metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
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
                    logits = model(x_batch_train, training=True)[0]  # Logits for this minibatch
                    active_loss = tf.reshape(y_batch_train, (-1,)) != -1
                    active_logits = tf.boolean_mask(tf.reshape(logits, (-1, tmp_sl_obj.num_labels)), active_loss)
                    active_labels = tf.boolean_mask(tf.reshape(y_batch_train, (-1,)), active_loss)
                    cross_entropy = loss_fn(active_labels, active_logits)
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                # Update training metric.
                train_metric(active_labels, active_logits)

                # Logging
                training_bar.update(train_step + 1,
                                    values=[('loss', float(loss)), ('acc', float(train_metric.result()))])

            # Reset training metrics at the end of each epoch
            train_metric.reset_states()
            validating_bar = tf.keras.utils.Progbar(valid_steps, stateful_metrics=['val_acc'])
            # Run a validation loop at the end of each epoch.
            for val_step, (x_batch_val, y_batch_val) in enumerate(valid_dataset):
                val_logits = model(x_batch_val, training=False)[0]
                active_loss_val = tf.reshape(y_batch_val, (-1,)) != -1
                active_logits_val = tf.boolean_mask(tf.reshape(val_logits, (-1, tmp_sl_obj.num_labels)), active_loss_val)
                active_labels_val = tf.boolean_mask(tf.reshape(y_batch_val, (-1,)), active_loss_val)
                # Update val metrics
                val_metric(active_labels_val, active_logits_val)
                # Logging
                validating_bar.update(val_step + 1, values=[('val_acc', float(val_metric.result()))])
            val_acc = val_metric.result()
            # Save the best model with the highest verification accuracy
            if val_acc > best_val_acc:
                print('model saving...')
                # normal
                model.save_pretrained(self.saved_model_path)
                best_val_acc = val_acc
            val_metric.reset_states()
        return model

    # todo
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

    # todo
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
        tmp_pred = trained_model.predict(inputs, batch_size=1024)
        if self.mode == 'binary':
            tmp_result = tf.nn.sigmoid(tmp_pred)
            return np.squeeze(tmp_result.numpy(), axis=-1)
        else:
            tmp_result = tf.nn.softmax(tmp_pred, axis=-1)
            return tmp_result.numpy()


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/ner')
    tmp_saved_model_path = os.path.join(ROOT_PATH, 'saved_models/ner/chinese-rbt3-01')
    tmp_sl_obj = SequenceLabeling('chinese-rbt3', os.path.join(raw_data_path, 'labels.txt'), 128,
                                  saved_model_path=tmp_saved_model_path)
    #### training step ####
    tmp_sl_obj.train_op(raw_data_path, epochs=2, batch_size=16)

    print('done.')
