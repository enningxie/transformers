# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
""" data processors and helpers """

import logging
import os
import pickle
import pandas as pd

from .file_utils import is_tf_available
from .data_utils import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        return_tensors=None,
        save_label_map_path=None
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum examples length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
        return_tensors
        save_label_map_path

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    if save_label_map_path:
        label_map_reverse = {tmp_value: tmp_key for tmp_key, tmp_value in label_map.items()}
        with open(save_label_map_path, 'wb') as f:
            pickle.dump(label_map_reverse, f, -1)
        logger.info(f"Saved label map to '{save_label_map_path}'.")
    len_examples = len(examples)
    all_inputs = []
    batch_length = -1
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing examples %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, max_length=max_length)
        input_ids = inputs["input_ids"]
        all_inputs.append(inputs)
        if len(input_ids) > batch_length:
            batch_length = len(input_ids)

    # padding part
    features = []
    for (ex_index, (tmp_inputs, example)) in enumerate(zip(all_inputs, examples)):
        input_ids = tmp_inputs["input_ids"]
        token_type_ids = tmp_inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = tmp_inputs["attention_mask"]
        # Zero-pad up to the sequence length.
        padding_length = batch_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
            len(input_ids), batch_length
        )
        assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
            len(attention_mask), batch_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise ValueError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    if return_tensors is None:
        return features
    elif return_tensors == "tf":
        if not is_tf_available():
            raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        dataset = tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )
        return dataset


class SequencePairClassificationProcessor(DataProcessor):
    """Processor for the Sequence Pair Classification data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]  # sentence1
            text_b = line[1]  # sentence2
            label = line[2]  # label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SingleSentenceClassificationProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self):
        self.labels = None

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy())
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # todo Just for test
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir=None):
        """See base class."""
        if data_dir is None:
            return self.labels
        else:
            return sorted(pd.read_csv(os.path.join(data_dir, "train.tsv"), header=0, sep='\t',
                                      dtype={'label': str}).label.unique())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        added_labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[0]  # sentence
            label = line[1]  # label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        if set_type == 'train':
            self.labels = sorted(list(added_labels))
        return examples


processors = {
    "spc": SequencePairClassificationProcessor,
    "ssc": SingleSentenceClassificationProcessor
}

output_modes = {
    "spc": "classification",
    "ssc": "classification"
}
