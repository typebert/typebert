# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library to process data for tagging task such as NER/POS."""
import collections
import os

from absl import logging
import tensorflow as tf
import json
from official.nlp.data import classifier_data_lib
import pdb
import numpy as np
from itertools import groupby
import statistics
import pdb
from math import sqrt
import multiprocessing as mp
import json
# from joblib import Parallel, delayed
# A negative label id for the padding label, which will not contribute
# to loss/metrics in training.
_PADDING_LABEL_ID = -1

# The special unknown token, used to substitute a word which has too many
# subwords after tokenization.
_UNK_TOKEN = "[UNK]"

projects= {}
files={}


class InputExample(object):
  """A single training/test example for token classification."""

  def __init__(self, sentence_id, words=None, label_ids=None, best_context=None, filename=None,
               project = None, user_defs = None, user_def_toks=None, user_def_ind=None, is_user_def=None):
    """Constructs an InputExample."""
    self.sentence_id = sentence_id
    self.words = words if words else []
    self.label_ids = label_ids if label_ids else []
    self.best_context = best_context if best_context else []
    self.filename = filename
    self.project = project
    self.user_defs = user_defs if user_defs else []
    self.user_def_toks = user_def_toks if user_def_toks else []
    self.user_def_ind = user_def_ind if user_def_ind else []
    self.is_user_def = is_user_def if is_user_def else []

  def add_word_and_label_id(self, word, label_id, is_user_def):
    """Adds word and label_id pair in the example."""
    self.words.append(word)
    self.label_ids.append(label_id)
    self.is_user_def.append(is_user_def)
    # if best_context is not None:
    #   self.best_context.append(best_context)


class InputUserExample(object):
  """A single training/test example for token classification."""

  def __init__(self, user_def=None, user_def_loc=None, tokens=None, project=None):
    """Constructs an InputExample."""
    self.user_def = user_def if user_def else []
    self.user_def_loc = user_def_loc if user_def_loc else []
    self.tokens = tokens if tokens else []
    self.project = project


  def add_word_and_label_id(self, tok, user_def_id):
    """Adds word and label_id pair in the example."""
    self.tokens.append(tok)
    self.user_def.append(user_def_id)



class InputUserDef(object):
    """A single training/test example for token classification."""
    def __init__(self, user_def=None, user_def_loc=None, user_def_sequence=None, tokens=None, user_def_string=None, project=None):
      """Constructs an InputExample."""
      self.user_def = user_def
      self.user_def_string = user_def_string
      self.user_def_loc = user_def_loc
      self.user_def_sequence = user_def_sequence if user_def_sequence else None
      self.tokens = tokens if tokens else None
      self.project = project


def _read_one_file(file_name, label_list):
  """Reads one file and returns a list of `InputExample` instances."""
  lines = tf.io.gfile.GFile(file_name, "r").readlines()
  examples = []
  label_id_map = {label: i for i, label in enumerate(label_list)}
  sentence_id = 0
  global projects
  for line in lines:
    # if line['f_name'].split('/')[0] in projects:
    #   project = projects[line['f_name'].split('/')[0]]
    # else:
    #   project = len(projects)
    #   projects[line['f_name'].split('/')[0]] = len(projects)
    project=None

    example = InputExample(sentence_id=0, filename=line['filename'], project = project)
    line = line.strip("\n")
    if line:
      # The format is: <token>\t<label> for train/dev set and <token> for test.
      items = line.split("\t")
      assert len(items) == 2 or len(items) == 1
      token = items[0].strip()

      # Assign a dummy label_id for test set
      label_id = label_id_map[items[1].strip()] if len(items) == 2 else 0
      example.add_word_and_label_id(token, label_id)
    else:
      # Empty line indicates a new sentence.
      if example.words:
        examples.append(example)
        sentence_id += 1
        example = InputExample(sentence_id=sentence_id, filename=line['filename'], project = project)

  if example.words:
    examples.append(example)
  return examples

def _read_type_file(file_name, label_list, user_defs_f , test=False):
  """Reads one file and returns a list of `InputExample` instances."""
  lines = load_jsonl(file_name)
  examples = []
  user_definitions = []
  label_id_map = {label: i for i, label in enumerate(label_list)}
  user_defs_map = {user_def: i for i, user_def in enumerate(user_defs_f)}
  sentence_id = 0
  global projects, files
  for line in lines:
    if line['prog_id'] not in projects:
      projects[line['prog_id']] = len(projects)
    project = projects[line['prog_id']]
    user_defs = []
    user_defs_string = []
    user_def_toks = []

    #define project level user defs
    if not line['filedata']:
      continue

    for _, contents in line['filedata'].items():
      for k, v in contents['user_map'].items():
        # if user_def_map[k] in user_defs:
        #   aind = user_defs.index(user_def_map[k])
        #   user_defs.pop(aind)
        #   user_def_toks.pop(aind)
        user_defs.append(user_defs_map[k])
        user_defs_string.append(k)
        user_def_toks.append(v)

    #operate on file level
    example = None
    for file, contents in line['filedata'].items():
      if file not in files:
        files[file] = len(files)
      file_id = files[file]


      for k, v in contents['user_map'].items():
        if user_defs_map[k] in user_defs:
          indices = [i for i, x in enumerate(user_defs) if x == user_defs_map[k]]
          for index in sorted(indices, reverse=True):
            del user_defs[index]
            del user_def_toks[index]
        user_defs.append(user_defs_map[k])
        user_def_toks.append(v)

      #add mapping for keys in user_def token sequence
        # if a_user_def not in user_def_map:
        #   user_def_map[a_user_def] = len(user_def_map)
        # user_defs.append(user_def_map[a_user_def])

      if test:
        locations = {ann['ty']['loc']: ann['ty']  for ann in contents['annotations'] if ann['category'] != 'Inferred' }
      else:
        locations = {ann['ty']['loc']: ann['ty'] for ann in contents['annotations']}

      example = InputExample(sentence_id=0, filename = file_id, project = project)
      user_def_labels_locations = []
      user_def_labels = []
      for ix, token in enumerate(contents['tokens']):
        # s_ix = str(ix)
        label_id = -1
        is_user_def_project = -1
        if ix in locations:
            # c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F
            t_ = locations[ix]['type'].replace('\n', ' ').replace('\r', '').replace('\t', '')
            if t_ != "any":
              if t_ in user_defs_string:
                user_def_labels_locations.append(ix)
                user_def_labels.append(user_defs_map[t_])
                is_user_def_project = 1
                #user_def_ix = user_defs.index(user_defs_map[t_]) if user_defs_map[t_] in user_defs else _PADDING_LABEL_ID
                # user_definitions.append(InputUserDef(user_def=user_defs_map[t_], user_def_loc=ix, user_def_sequence=user_def_toks[user_def_ix], tokens=contents['tokens'], user_def_string=t_, project=project))

              label_id = label_id_map[t_] if t_ in label_id_map else label_id_map['UNK']

        else:
            label_id = _PADDING_LABEL_ID


        # if is_user_def_project ==1 and label_id!=0:
        #   import pdb
        #   pdb.set_trace()
        example.add_word_and_label_id(token, label_id, is_user_def_project)

      if example.words:
        examples.append(example)
        sentence_id += 1
        example = InputExample(sentence_id=sentence_id, filename = file_id, project = project)

      if user_def_labels:
        user_definitions.append(
          InputUserDef(user_def=user_def_labels, user_def_loc=user_def_labels_locations, tokens=contents['tokens'], project=project))

    # try:
    if example and example.words:
      examples.append(example)
    # except Exception as e:
    #   import pdb
    #   pdb.set_trace()

  return examples, user_definitions


class PanxProcessor(classifier_data_lib.DataProcessor):
  """Processor for the Panx data set."""
  supported_languages = [
      "ar", "he", "vi", "id", "jv", "ms", "tl", "eu", "ml", "ta", "te", "af",
      "nl", "en", "de", "el", "bn", "hi", "mr", "ur", "fa", "fr", "it", "pt",
      "es", "bg", "ru", "ja", "ka", "ko", "th", "sw", "yo", "my", "zh", "kk",
      "tr", "et", "fi", "hu"
  ]

  def get_train_examples(self, data_dir):
    return _read_one_file(
        os.path.join(data_dir, "train-en.tsv"), self.get_labels())

  def get_dev_examples(self, data_dir):
    return _read_one_file(
        os.path.join(data_dir, "dev-en.tsv"), self.get_labels())

  def get_test_examples(self, data_dir):
    examples_dict = {}
    for language in self.supported_languages:
      examples_dict[language] = _read_one_file(
          os.path.join(data_dir, "test-%s.tsv" % language), self.get_labels())
    return examples_dict

  def get_labels(self):
    return ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

  @staticmethod
  def get_processor_name():
    return "panx"


class UdposProcessor(classifier_data_lib.DataProcessor):
  """Processor for the Udpos data set."""
  supported_languages = [
      "af", "ar", "bg", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fr",
      "he", "hi", "hu", "id", "it", "ja", "kk", "ko", "mr", "nl", "pt", "ru",
      "ta", "te", "th", "tl", "tr", "ur", "vi", "yo", "zh"
  ]

  def get_train_examples(self, data_dir):
    return _read_one_file(
        os.path.join(data_dir, "train-en.tsv"), self.get_labels())

  def get_dev_examples(self, data_dir):
    return _read_one_file(
        os.path.join(data_dir, "dev-en.tsv"), self.get_labels())

  def get_test_examples(self, data_dir):
    examples_dict = {}
    for language in self.supported_languages:
      examples_dict[language] = _read_one_file(
          os.path.join(data_dir, "test-%s.tsv" % language), self.get_labels())
    return examples_dict

  def get_labels(self):
    return [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
    ]

  @staticmethod
  def get_processor_name():
    return "udpos"


class TsTypeProcessor(classifier_data_lib.DataProcessor):
  """Processor for the TsType data set."""

  def get_train_examples(self, data_dir):
    return _read_type_file(os.path.join(data_dir, "train.jsonl"), self.get_labels(data_dir), self.get_user_defs(data_dir))

  def get_dev_examples(self, data_dir):
    return _read_type_file(os.path.join(data_dir, "dev.jsonl"), self.get_labels(data_dir), self.get_user_defs(data_dir))

  def get_test_examples(self, data_dir):
    return _read_type_file(os.path.join(data_dir, "test.jsonl"), self.get_labels(data_dir), self.get_user_defs(data_dir), test=True)

  def get_labels(self, data_dir):
      with open(os.path.join(data_dir, "labels.txt")) as f:
          return f.read().splitlines()

  def get_user_defs(self, data_dir):
      with open(os.path.join(data_dir, "user_defs.txt")) as f:
          return f.read().splitlines()

  @staticmethod
  def get_processor_name():
    return "ts_type"

def _tokenize_user_def(user_def, max_length, tokenizer, text_preprocessing=None):
  """Tokenizes words and breaks long example into short ones."""
  # Needs additional [CLS] and [SEP] tokens.

  max_length = max_length - 2
  new_examples = []

  offset = 0
  new_example = InputUserExample(project=user_def.project)
  for i, word in enumerate(user_def.tokens):
    if text_preprocessing:
      word = text_preprocessing(word)
    subwords = tokenizer.tokenize(word)
    if word and (not subwords or len(subwords) > max_length):
      subwords = [_UNK_TOKEN]

    if len(new_example.tokens) + len(subwords) > max_length:
      if new_example.user_def and new_example.user_def.count(_PADDING_LABEL_ID) != len(new_example.user_def):
        new_examples.append(new_example)
      new_example = InputUserExample(project=user_def.project)
      offset = 0

    for j, subword in enumerate(subwords):
      new_example.tokens.append(subword)
      if j == 0 and i in user_def.user_def_loc:
        lab = user_def.user_def.pop(0)
        # user_def.user_def_loc.pop(0)
        new_example.user_def.append(lab)
      else:
        new_example.user_def.append(_PADDING_LABEL_ID)
    offset += len(subwords)

  if new_example.tokens and new_example.user_def and new_example.user_def.count(_PADDING_LABEL_ID) != len(new_example.user_def) :
    new_examples.append(new_example)



  return new_examples

def _tokenize_example(example, max_length, tokenizer, text_preprocessing=None, use_neg_labels=True, doc_stride=None):
  """Tokenizes words and breaks long example into short ones."""
  # Needs additional [CLS] and [SEP] tokens.
  max_length = max_length - 2
  new_examples = []
  new_example = InputExample(sentence_id=example.sentence_id, filename=example.filename, project=example.project)
  for i, word in enumerate(example.words):
    if not use_neg_labels and any([x < 0 for x in example.label_ids]):
      raise ValueError("Unexpected negative label_id: %s" % example.label_ids)

    if text_preprocessing:
      word = text_preprocessing(word)
    subwords = tokenizer.tokenize(word)
    if (not subwords or len(subwords) > max_length) and word:
      subwords = [_UNK_TOKEN]

    if len(subwords) + len(new_example.words) > max_length:
      # Start a new example. Only add if there is a label that is not all -1
      if new_example.label_ids.count(_PADDING_LABEL_ID) != len(new_example.label_ids) and new_example.label_ids:
        new_examples.append(new_example)
      new_example = InputExample(sentence_id=example.sentence_id, filename=example.filename, project=example.project)

    for j, subword in enumerate(subwords):
      # Use the real label for the first subword, and pad label for
      # the remainings.
      subword_label = example.label_ids[i] if j == 0 else _PADDING_LABEL_ID
      is_user_def_subword = example.is_user_def[i] if j == 0 else _PADDING_LABEL_ID
      # best_context = 1 if subword_label!=_PADDING_LABEL_ID else 0

      new_example.add_word_and_label_id(subword, subword_label, is_user_def_subword)

  if new_example.words and new_example.label_ids.count(_PADDING_LABEL_ID) != len(new_example.label_ids) and new_example.label_ids:
    new_examples.append(new_example)

  return new_examples

def _convert_single_user_def(example, max_seq_length, tokenizer):
  tokens = ["[CLS]"]
  tokens.extend(example.tokens)
  tokens.append("[SEP]")
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  label_ids = [_PADDING_LABEL_ID]
  label_ids.extend(example.user_def)
  label_ids.append(_PADDING_LABEL_ID)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    label_ids.append(_PADDING_LABEL_ID)

  # user_def_seq = ["[CLS]"]
  # user_def_seq.extend(example.user_def_sequence)
  # user_def_seq.append("[SEP]")
  # user_def_seq_ids = tokenizer.convert_tokens_to_ids(user_def_seq)
  # while len(user_def_seq_ids) < max_seq_length:
  #   user_def_seq_ids.append(0)

  if len(input_ids) > max_seq_length or len(label_ids) > max_seq_length:
    print("BUG")
    pdb.set_trace()

  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  # features["user_def_id"] = create_int_feature(user_def_seq_ids)
  #features["user_def_loc"]= create_int_feature([example.user_def_loc+1]) #we add one because the CLS token shifts the indexes
  features["user_def"] = create_int_feature(label_ids)
  features["project"] = create_int_feature([example.project])
  # features["user_defs"] = create_int_feature(example.user_defs)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example


  # new_examples.append(InputUserDef(user_def=user_def.user_def, user_def_loc=new_loc, user_def_sequence=e,
  #                                  tokens=mytokens, user_def_string=user_def.user_def_string, project=user_def.project))

  pass


def _convert_single_example(example, max_seq_length, tokenizer):
  """Converts an `InputExample` instance to a `tf.train.Example` instance."""
  tokens = ["[CLS]"]
  tokens.extend(example.words)
  tokens.append("[SEP]")
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  label_ids = [_PADDING_LABEL_ID]
  label_ids.extend(example.label_ids)
  label_ids.append(_PADDING_LABEL_ID)

  is_user_def = [_PADDING_LABEL_ID]
  is_user_def.extend(example.is_user_def)
  is_user_def.append(_PADDING_LABEL_ID)

  # context = [False]+example.best_context+[False]

  segment_ids = [0] * len(input_ids)
  input_mask = [1] * len(input_ids)
  context = [0]* len(input_ids)

  # Pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(_PADDING_LABEL_ID)
    is_user_def.append(_PADDING_LABEL_ID)
    context.append(0)


  def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  features["label_ids"] = create_int_feature(label_ids)
  features["sentence_id"] = create_int_feature([example.sentence_id])
  features["best_context"]= create_int_feature(context)
  features["filename"] = create_int_feature([example.filename])
  features["project"] = create_int_feature([example.project])
  features["is_user_def"] = create_int_feature(is_user_def)
  # features["user_defs"] = create_int_feature(example.user_defs)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None

  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue

    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def convert_examples_to_features(example, tokenizer, max_seq_length, doc_stride, is_training=False):
  """Loads a data file into a list of `InputBatch`s."""

  base_id = 1000000000
  unique_id = base_id
  all_examples = []
  # for (example_index, example) in enumerate(examples):

  all_doc_tokens = []
  all_doc_labels = []
  example_label_ix = []

  # from time import time
  # st = time()
  for i, (token, label) in enumerate(zip(example.words, example.label_ids)):

    sub_tokens = tokenizer.tokenize(token)

    if not sub_tokens and token:
      sub_tokens = [_UNK_TOKEN]

    for sub_token in sub_tokens:
      all_doc_tokens.append(sub_token)

    for j, sub_token in enumerate(sub_tokens):
      sublabel = label if j == 0 else _PADDING_LABEL_ID
      if sublabel!=_PADDING_LABEL_ID:
          example_label_ix.append(len(all_doc_labels))
      all_doc_labels.append(sublabel)

  # The -2 accounts for [CLS], [SEP]
  max_tokens_for_doc = max_seq_length - 2

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of the up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length", "labels", "best_context"])
  doc_spans = []

  start_offset = 0

  #create all the doc spans for the document
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    mask = [True if ix >=start_offset and ix<(start_offset+length) else False for ix in example_label_ix]
    in_span = any(mask)
    span_labels = tuple(np.array(example_label_ix)[mask].tolist())

    if in_span:
      doc_spans.append(_DocSpan(start=start_offset, length=length, labels=span_labels, best_context=[False]*length)) #is this a problem
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, doc_stride)

  #remove any subset examples i.e DocSpan(start=6218, length=126, labels=(6256, 6281, 6311)) and
  # DocSpan(start=6248, length=126, labels=(6256, 6281, 6311, 6359, 6363))
  # indices_to_delete = {}
  # for si, s_1 in enumerate(doc_spans):
  #     s1_s = set(s_1.labels)
  #     s2_copy = doc_spans[:si]+doc_spans[si+1:]
  #     for sj, s_2 in enumerate(s2_copy):
  #         s2_s = set(s_2.labels)
  #         if s1_s <= s2_s:
  #             indices_to_delete.add(si)
  #
  # for index in sorted(indices_to_delete, reverse=True):
  #     del doc_spans[index]


  # optimal_span_list = []
  # for v in example_label_ix:  # start adding the context from document
  #   for (doc_span_index, doc_span) in enumerate(doc_spans):
  #
  #     if v < doc_span.start or v >= doc_span.start+ doc_span.length:
  #       continue
  #     # split_token_index = doc_span.start + i
  #     is_max_context = _check_is_max_context(doc_spans, doc_span_index,
  #                                            v)
  #
  #     if is_max_context:
  #       doc_span.best_context[v-doc_span.start] = 1
  #       optimal_span_list.append(doc_span)

  optimal_span_list = []

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    has_is_max_context = False
    for v in example_label_ix:  # start adding the context from document
      if v < doc_span.start or v >= doc_span.start+ doc_span.length:
        continue
      # split_token_index = doc_span.start + i
      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             v)

      if is_max_context:
        has_is_max_context = True
        doc_span.best_context[v-doc_span.start] = True
    if has_is_max_context:
      optimal_span_list.append(doc_span)


  for span in optimal_span_list:
    tokens = all_doc_tokens[span.start:span.start + span.length]
    labels = all_doc_labels[span.start:span.start + span.length]
    all_examples.append(
      InputExample(example.sentence_id, words=tokens, label_ids=labels, best_context=span.best_context, filename=example.filename, project=example.project))

  # # for eval
  # is_training=False
  # if not is_training:
  #   for span in optimal_span_list:
  #     tokens = all_doc_tokens[span.start:span.start+span.length]
  #     labels = all_doc_labels[span.start:span.start+span.length]
  #     all_examples.append(InputExample(example.sentence_id, words=tokens, label_ids=labels, best_context=span.best_context))
  # else:
  #
  #   # only do this for training.
  #   for k, v in groupby(optimal_span_list, lambda x: x.labels):
  #     s, e, _, bc = list(zip(*list(v)))
  #     e_n = [x + y for x, y in zip(s, e)]
  #     # label_ind = [i for i, x in enumerate(all_doc_labels) if x == 0]
  #     #
  #     # start = int(statistics.mean(s))
  #     # end = int(statistics.mean(e_n))
  #     # tokens = all_doc_tokens[start:end]
  #     # labels = all_doc_labels[start:end]
  #     pdb.set_trace()
  #
  #     min_left_st = min(s)
  #     max_right_end = max(e_n)
  #     label_ind = list(k)
  #     min_label = min(label_ind)
  #     max_label = max(label_ind)
  #
  #     # if training make the best possible window
  #     i = min_label
  #     j = max_label
  #
  #
  #
  #
  #     while ((i > min_left_st or j < max_right_end) and (j - i) < max_seq_length - 2):
  #       # if j == max_right_end and i == min_left_st:
  #       #   break
  #       if i > min_left_st:
  #         i -= 1
  #       if j < max_right_end:
  #         j += 1
  #
  #     start = i
  #     end = j
  #
  #     best_context = [False] * (j-i)
  #     # pdb.set_trace()
  #
  #     for i in label_ind:
  #       best_context[i - start] = True
  #     tokens = all_doc_tokens[start:end]
  #     labels = all_doc_labels[start:end]
  #     all_examples.append(InputExample(example.sentence_id, words=tokens, label_ids=labels, best_context=best_context))
  # # print(time()-st)
  # # print(st)

  return all_examples


def write_user_defs_to_file(user_defs,
                          tokenizer,
                          max_seq_length,
                          output_file,
                          text_preprocessing=None,
                          doc_stride=None, is_training=False):

  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  def perform_convert_to_features(user_def, max_seq_length, tokenizer):
    tokenized_examples = _tokenize_user_def(user_def, max_seq_length, tokenizer,text_preprocessing)
    for per_tokenized_example in tokenized_examples:
      tf_example = _convert_single_user_def(per_tokenized_example, max_seq_length, tokenizer)
      writer.write(tf_example.SerializeToString())
    return len(tokenized_examples)

  num_tokenized_examples = 0
  for i, example in enumerate(user_defs):
    print(i)
    tok_examples = perform_convert_to_features(example, max_seq_length, tokenizer)
    num_tokenized_examples+=tok_examples

  writer.close()
  return num_tokenized_examples

def write_example_to_file(examples,
                          tokenizer,
                          max_seq_length,
                          output_file,
                          text_preprocessing=None,
                          doc_stride=None, is_training=False):
  """Writes `InputExample`s into a tfrecord file with `tf.train.Example` protos.

  Note that the words inside each example will be tokenized and be applied by
  `text_preprocessing` if available. Also, if the length of sentence (plus
  special [CLS] and [SEP] tokens) exceeds `max_seq_length`, the long sentence
  will be broken into multiple short examples. For example:

  Example (text_preprocessing=lowercase, max_seq_length=5)
    words:        ["What", "a", "great", "weekend"]
    labels:       [     7,   5,       9,        10]
    sentence_id:  0
    preprocessed: ["what", "a", "great", "weekend"]
    tokenized:    ["what", "a", "great", "week", "##end"]

  will result in two tf.example protos:

    tokens:      ["[CLS]", "what", "a", "great", "[SEP]"]
    label_ids:   [-1,       7,     5,     9,     -1]
    input_mask:  [ 1,       1,     1,     1,      1]
    segment_ids: [ 0,       0,     0,     0,      0]
    input_ids:   [ tokenizer.convert_tokens_to_ids(tokens) ]
    sentence_id: 0

    tokens:      ["[CLS]", "week", "##end", "[SEP]", "[PAD]"]
    label_ids:   [-1,       10,     -1,    -1,       -1]
    input_mask:  [ 1,       1,       1,     0,        0]
    segment_ids: [ 0,       0,       0,     0,        0]
    input_ids:   [ tokenizer.convert_tokens_to_ids(tokens) ]
    sentence_id: 0

    Note the use of -1 in `label_ids` to indicate that a token should not be
    considered for classification (e.g., trailing ## wordpieces or special
    token). Token classification models should accordingly ignore these when
    calculating loss, metrics, etc...

  Args:
    examples: A list of `InputExample` instances.
    tokenizer: The tokenizer to be applied on the data.
    max_seq_length: Maximum length of generated sequences.
    output_file: The name of the output tfrecord file.
    text_preprocessing: optional preprocessing run on each word prior to
      tokenization.

  Returns:
    The total number of tf.train.Example proto written to file.
  """
  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  def count_labels(example):
    return len(example.label_ids) - example.label_ids.count(_PADDING_LABEL_ID)

  examples_n = []
  for example in examples:
    if count_labels(example) != 0:
      examples_n.append(example)

  #sort examples by label_ids
  examples_s = sorted(examples_n, key=count_labels)

  def perform_convert_to_features(example, tokenizer,max_seq_length,doc_stride):
    if doc_stride:
        tokenized_examples = convert_examples_to_features(example, tokenizer=tokenizer,max_seq_length=max_seq_length, doc_stride=doc_stride, is_training=is_training)
    else:
        tokenized_examples = _tokenize_example(example, max_seq_length, tokenizer,text_preprocessing, doc_stride=None)


    for per_tokenized_example in tokenized_examples:
      tf_example = _convert_single_example(per_tokenized_example,
                                           max_seq_length, tokenizer)
      writer.write(tf_example.SerializeToString())
    return len(tokenized_examples)

  num_tokenized_examples = 0
  for i, example in enumerate(examples_s):
    num_examples = perform_convert_to_features(example, tokenizer, max_seq_length, doc_stride)
    num_tokenized_examples+=num_examples
  writer.close()
  return num_tokenized_examples


def token_classification_meta_data(train_data_size,
                                   max_seq_length,
                                   num_labels,
                                   eval_data_size=None,
                                   test_data_size=None,
                                   label_list=None,
                                   processor_type=None):
  """Creates metadata for tagging (token classification) datasets."""
  meta_data = {
      "train_data_size": train_data_size,
      "max_seq_length": max_seq_length,
      "num_labels": num_labels,
      "task_type": "tagging",
      "label_type": "int",
      "label_shape": [max_seq_length],
  }
  if eval_data_size:
    meta_data["eval_data_size"] = eval_data_size
  if test_data_size:
    meta_data["test_data_size"] = test_data_size
  if label_list:
    meta_data["label_list"] = label_list
  if processor_type:
    meta_data["processor_type"] = processor_type

  return meta_data

def user_embeddings_meta_data(train_data_size,
                                   max_seq_length,
                                   eval_data_size=None,
                                   test_data_size=None,
                                   user_def_list=None
):
  """Creates metadata for tagging (token classification) datasets."""
  meta_data = {
      "train_data_size": train_data_size,
      "test_data_size": test_data_size,
      "max_seq_length": max_seq_length,
      "task_type": "embeddings",
  }
  if eval_data_size:
    meta_data["eval_data_size"] = eval_data_size
  if test_data_size:
    meta_data["test_data_size"] = test_data_size
  if user_def_list:
    meta_data["user_def_list"] = user_def_list

  return meta_data


def generate_tf_record_from_data_file(processor, data_dir, tokenizer,
                                      max_seq_length, train_data_output_path,
                                      eval_data_output_path,
                                      test_data_output_path,
                                      text_preprocessing, doc_stride, user_train_data_output_path,
                                      user_test_data_output_path, ):
  """Generates tfrecord files from the raw data."""
  common_kwargs = dict(
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      text_preprocessing=text_preprocessing,
      doc_stride=doc_stride)

  print("DOC_STRIDE")
  print(doc_stride)
  doc_stride = None

  # eval_examples = processor.get_dev_examples(data_dir)
  # eval_data_size = write_example_to_file(
  #     eval_examples, output_file=eval_data_output_path, **common_kwargs)

  test_examples, test_user_defs = processor.get_test_examples(data_dir)
  # print("user_defs")
  # test_user_data_size = write_user_defs_to_file(test_user_defs, output_file=user_test_data_output_path, **common_kwargs)
  print("type examples")
  test_data_size = write_example_to_file(test_examples, output_file=test_data_output_path, **common_kwargs)


  global files
  import json
  json = json.dumps(files)
  f = open(data_dir+"/file_map_dedup.json", "w")
  f.write(json)
  f.close()


  #
  # train_examples, train_user_defs = processor.get_train_examples(data_dir)
  # print("user_defs")
  # train_user_data_size = write_user_defs_to_file(train_user_defs, output_file=user_train_data_output_path, **common_kwargs)
  # print("type examples")
  # train_data_size = write_example_to_file(train_examples, output_file=train_data_output_path, **common_kwargs, is_training=True)

  # train_user_data_size = train_data_size = 0
  # eval_user_data_size = test_user_data_size=0
  # train_data_size= eval_data_size = test_data_size

  # labels = processor.get_labels(data_dir)
  # user_definition_list = processor.get_user_defs(data_dir)
  # meta_data = token_classification_meta_data(
  #     train_data_size,
  #     max_seq_length,
  #     len(labels),
  #     eval_data_size,
  #     test_data_size,
  #     label_list=labels,
  #     processor_type=processor.get_processor_name())
  #
  # user_meta_data = user_embeddings_meta_data(train_user_data_size,
  #                                            max_seq_length,
  #                                            eval_user_data_size,
  #                                            test_user_data_size,
  #                                            user_definition_list)

  pdb.set_trace()

  return meta_data, user_meta_data


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

