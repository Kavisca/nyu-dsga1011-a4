import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
_TOKENIZER_CACHE = {}


def build_schema_vocab(data_folder="data"):
    '''
    Collect SQL identifiers containing an underscore (table/column names, aliases like
    airport_service_1, and the greater_than/less_than placeholders) from the training SQL.
    Adding these as whole tokens roughly halves the decoder sequence length. Only train.sql
    is used and the list is sorted, so the vocabulary is deterministic across runs (the
    saved checkpoint's embedding size depends on it).
    '''
    vocab = set()
    for query in load_lines(os.path.join(data_folder, "train.sql")):
        query = re.sub(r"'[^']*'", " ", preprocess_sql(query, lowercase=True))
        for word in re.findall(r"[a-z_][a-z0-9_]*", query):
            if "_" in word:
                vocab.add(word)
    return sorted(vocab)


def get_tokenizer(schema_tokens=False, data_folder="data"):
    key = (schema_tokens, data_folder)
    if key not in _TOKENIZER_CACHE:
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        if schema_tokens:
            tokenizer.add_tokens(build_schema_vocab(data_folder))
        _TOKENIZER_CACHE[key] = tokenizer
    return _TOKENIZER_CACHE[key]


def decode_sql(tokenizer, ids):
    '''
    Decode generated ids. The fast tokenizer concatenates added tokens without spaces
    ("flightflight_1"), so added tokens are emitted with surrounding spaces and the
    remaining pieces are decoded normally. postprocess_sql then normalizes whitespace.
    '''
    ids = [int(i) for i in ids]
    added = tokenizer.get_added_vocab()
    added_ids = {i: tok for tok, i in added.items() if not tok.startswith("<")}
    special = set(tokenizer.all_special_ids)
    pieces, run = [], []
    for i in ids:
        if i in added_ids:
            if run:
                pieces.append(tokenizer.decode(run, skip_special_tokens=True))
                run = []
            pieces.append(f" {added_ids[i]} ")
        elif i not in special:
            run.append(i)
    if run:
        pieces.append(tokenizer.decode(run, skip_special_tokens=True))
    return "".join(pieces)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split, schema_tokens=False, lowercase_sql=False):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.lowercase_sql = lowercase_sql
        self.tokenizer = get_tokenizer(schema_tokens, data_folder)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        encoder_text = load_lines(nl_path)

        encoder_ids = tokenizer(
            encoder_text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )["input_ids"]

        initial_decoder_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

        if split == "test":
            processed = []
            for enc_ids in encoder_ids:
                processed.append((torch.tensor(enc_ids, dtype=torch.long),
                                  torch.tensor([initial_decoder_id], dtype=torch.long)))
            return processed

        sql_path = os.path.join(data_folder, f"{split}.sql")
        decoder_text = load_lines(sql_path)
        if len(encoder_text) != len(decoder_text):
            raise ValueError(f"{split}: natural-language and SQL files have different lengths")

        decoder_text = [preprocess_sql(query, lowercase=self.lowercase_sql) for query in decoder_text]
        target_ids = tokenizer(
            decoder_text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )["input_ids"]

        processed = []
        for enc_ids, target in zip(encoder_ids, target_ids):
            target = torch.tensor(target, dtype=torch.long)
            decoder_input = torch.cat([
                torch.tensor([initial_decoder_id], dtype=torch.long),
                target[:-1],
            ])
            processed.append((torch.tensor(enc_ids, dtype=torch.long),
                              decoder_input,
                              target,
                              torch.tensor([initial_decoder_id], dtype=torch.long)))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack([item[3] for item in batch])

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack([item[1] for item in batch])
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, schema_tokens=False, lowercase_sql=False):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, schema_tokens=schema_tokens, lowercase_sql=lowercase_sql)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, schema_tokens=False, lowercase_sql=False):
    train_loader = get_dataloader(batch_size, "train", schema_tokens, lowercase_sql)
    dev_loader = get_dataloader(test_batch_size, "dev", schema_tokens, lowercase_sql)
    test_loader = get_dataloader(test_batch_size, "test", schema_tokens, lowercase_sql)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x

SQL_OPERATORS = [(">=", "GREATER_THAN_OR_EQUAL"), ("<=", "LESS_THAN_OR_EQUAL"),
                 (">", "GREATER_THAN"), ("<", "LESS_THAN")]  # longest first


def preprocess_sql(sql, lowercase=False):
    '''
    Fine-tuned model (lowercase=False): case preserved; >=, <=, >, < replaced by
    GREATER_THAN_OR_EQUAL, LESS_THAN_OR_EQUAL, GREATER_THAN, LESS_THAN.
    Scratch model (lowercase=True): the whole query is lowercased and the operators
    become greater_than_or_equal, etc.
    '''
    if lowercase:
        sql = sql.lower()
    for op, word in SQL_OPERATORS:
        sql = sql.replace(op, f" {word.lower() if lowercase else word} ")
    return re.sub(r"\s+", " ", sql).strip()


def postprocess_sql(sql, lowercase=False):
    '''
    Reverse preprocess_sql (longest operator first) and normalize whitespace.
    For the scratch model (lowercase=True), also uppercase string literals (database
    values are uppercase) and re-join "alias . column" split apart by schema-token decoding.
    '''
    for op, word in SQL_OPERATORS:
        sql = re.sub(word, op, sql, flags=re.I)
    sql = re.sub(r"\s+", " ", sql).strip()
    if lowercase:
        sql = re.sub(r"'[^']*'", lambda match: match.group(0).upper(), sql)
        sql = re.sub(r"(\w_\d+)\s*\.\s*(?=[a-z_])", r"\1.", sql)
    return sql