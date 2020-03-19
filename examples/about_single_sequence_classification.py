# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/18
"""Single sequence classification"""
import os
import logging
import pickle
import tensorflow as tf
from src.transformers.file_utils import ROOT_PATH, CONFIG_NAME, LABEL2ID_NAME, ID2CLASS_NAME
from src.transformers.configuration_bert import BertConfig
from src.transformers.tokenization_bert import BertTokenizer
from src.transformers.modeling_tf_bert import TFBertForSequenceClassification
from src.transformers.data_processors import SingleSentenceClassificationProcessor, convert_examples_to_features
from src.transformers.modeling_tf_utils import calculate_steps

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_serialized_data(data_dir):
    # 恢复数据
    with open(data_dir, 'rb') as file:
        data = pickle.load(file)
    return data


class AboutSingleSequenceClassification:
    """
    单句表述分类
    """

    def __init__(self, pretrained_model_name, num_labels, max_length):
        self.data_processor = SingleSentenceClassificationProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.num_labels = num_labels
        self.pretrained_model_name = pretrained_model_name

        # training parameters
        self.max_length = max_length

        self.task = 'ssc'

    def generate_tf_dataset(self, data_path, batch_size):
        """
        生成TFDataSet，用于训练模型前的准备
        :param data_path:
        :return:
        """
        # process raw data
        train_examples = self.data_processor.get_train_examples(data_path)
        train_label_list = self.data_processor.get_labels()
        valid_examples = self.data_processor.get_dev_examples(data_path)
        valid_label_list = self.data_processor.get_labels()

        # calculate steps
        train_steps = calculate_steps(len(train_examples), batch_size)
        valid_steps = calculate_steps(len(valid_examples), batch_size)

        # convert examples to tf_dataset
        train_dataset = convert_examples_to_features(train_examples, self.tokenizer, max_length=self.max_length,
                                                     label_list=train_label_list, task=self.task, return_tensors='tf',
                                                     save_label_map_path=os.path.join(data_path, 'label2id.pkl'))
        valid_dataset = convert_examples_to_features(valid_examples, self.tokenizer, max_length=self.max_length,
                                                     label_list=valid_label_list, task=self.task, return_tensors='tf')

        # preprocess tf_dataset
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)

        return (train_dataset, train_steps), (valid_dataset, valid_steps)

    @staticmethod
    def get_trained_model(trained_model_path):
        """
        加载训练好的模型
        :param trained_model_path:
        :return:
        """
        trained_config = BertConfig.from_pretrained(os.path.join(trained_model_path, CONFIG_NAME))
        trained_model = TFBertForSequenceClassification.from_pretrained(trained_model_path, config=trained_config)
        return trained_model

    @staticmethod
    def get_compiled_model(model):
        """
        返回编译后的模型
        :param model:
        :return:
        """
        # prepare training: compile tf.keras model with optimizer, loss and learning rate schedule
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        model.summary()
        return model

    def train_op(self, data_path, epochs, batch_size, saved_model_path):
        """
        模型训练
        :param data_path: 原始数据保存路径
        :param epochs:
        :param batch_size:
        :param saved_model_path:
        :return:
        """
        config = BertConfig.from_pretrained(self.pretrained_model_name, num_labels=self.num_labels)
        model = TFBertForSequenceClassification.from_pretrained(self.pretrained_model_name, config=config)
        model = self.get_compiled_model(model)
        (train_dataset, train_steps), (valid_dataset, valid_steps) = self.generate_tf_dataset(data_path, batch_size)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

        # Train and evaluate using tf.keras.Model.fit()
        history = model.fit(train_dataset.repeat(), epochs=epochs, steps_per_epoch=train_steps,
                            validation_data=valid_dataset.repeat(), validation_steps=valid_steps,
                            callbacks=[early_stop])
        print(f"History: {history.history}")
        model.save_pretrained(saved_model_path)

    def evaluate_op(self, trained_model_path, test_data_path, batch_size=64):
        """
        模型 evaluation step
        :param trained_model_path:
        :param test_data_path:
        :param batch_size:
        :return:
        """
        # 模型加载
        trained_model = self.get_trained_model(trained_model_path)
        trained_model = self.get_compiled_model(trained_model)
        # 原始数据转tf-dataset
        test_examples = self.data_processor.get_test_examples(test_data_path)
        test_label_list = self.data_processor.get_labels()

        test_steps = calculate_steps(len(test_examples), batch_size)
        test_dataset = convert_examples_to_features(test_examples, self.tokenizer, max_length=self.max_length,
                                                    label_list=test_label_list, task=self.task, return_tensors='tf')
        test_dataset = test_dataset.batch(batch_size)
        # model evaluation step
        evaluation_loss, evaluation_acc = trained_model.evaluate(test_dataset, steps=test_steps)
        print(f"Evaluate step loss: {evaluation_loss}")
        print(f"Evaluate step accuracy: {evaluation_acc}")

    def predict_op(self, trained_model_path, batch_text):
        """
        :param trained_model_path:
        :param batch_text: [['想了解下您会想看哪款车型', '是想请问下您当时买的是哪款车呢'], ['今天天气很差', '今天天气很棒']]
        :return:
        """
        trained_model = self.get_trained_model(trained_model_path)
        inputs = self.tokenizer.batch_encode_plus(batch_text, max_length=self.max_length, return_tensors="tf",
                                                  pad_to_max_length=True)
        tmp_pred = trained_model.predict(inputs)
        tmp_pred_label = tf.argmax(tmp_pred, axis=-1).numpy()
        return tmp_pred_label


if __name__ == '__main__':
    raw_data_path = os.path.join(ROOT_PATH, 'data/souche_salesman')
    tmp_spc_obj = AboutSingleSequenceClassification('chinese-rbt3',
                                                    num_labels=25,
                                                    max_length=32)
    # #### training step ####
    # tmp_saved_model_path = os.path.join(ROOT_PATH, 'examples/saved_models/ssc_2')
    # tmp_spc_obj.train_op(raw_data_path, epochs=5, batch_size=64, saved_model_path=tmp_saved_model_path)

    tmp_trained_model_path = os.path.join(ROOT_PATH, 'examples/saved_models/ssc_2')

    # #### evaluate step ####
    # tmp_spc_obj.evaluate_op(tmp_trained_model_path, raw_data_path)

    #### predict step ####
    tmp_batch_text = ['有空可以到店里面先试驾一下啊', '最近没时间']
    tmp_pred_label = tmp_spc_obj.predict_op(tmp_trained_model_path, tmp_batch_text)

    tmp_label2id_path = os.path.join(raw_data_path, LABEL2ID_NAME)
    tmp_id2class_path = os.path.join(raw_data_path, ID2CLASS_NAME)

    label2id = load_serialized_data(tmp_label2id_path)
    id2class = load_serialized_data(tmp_id2class_path)

    print(label2id)
    print(id2class)

    for tmp_text, tmp_label in zip(tmp_batch_text, tmp_pred_label):
        print(f'{tmp_text} --> {id2class[int(label2id[tmp_label])]}')