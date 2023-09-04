import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Sequence
import pytorch_lightning as pl
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import transformers
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from typing import List, Union
from datasets import load_from_disk
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import copy
import torch
IGNORE_INDEX = -100
    
class BaseDataset(Dataset):
    def __init__(self, input, label, tokenizer, **kwargs):
        self.input = input
        self.label = label
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        assert len(self.input) == len(self.label)
    def __len__(self):
        assert len(self.input) == len(self.label)
        if self.kwargs['max_samples'] is not None:
            return self.kwargs['max_samples']
        else:
            return len(self.input)

    def __getitem__(self, idx):

        return {
            'input': self.input[idx],
            'label': self.label[idx]
        }
    
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['label']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict    
    
@dataclass
class BaseDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors


        input = [feature["input"] for feature in features]
        label = [feature["label"] for feature in features]
        ## for tokenizer's retrun type = 'pt'
        # input = {
        #     key: [example[key][0] for example in input]
        #     for key in input[0].keys()
        # }
        # label = {
        #     key: [example[key][0] for example in label]
        #     for key in label[0].keys()
        # }
        input = self.tokenizer.pad(
            input,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        label = self.tokenizer.pad(
            label,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        make_result = {}
        make_result["input"] = input
        make_result["label"] = label

        return make_result 
    
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len = self.args.source_max_len,
        target_max_len = self.args.target_max_len,
        train_on_source= self.args.train_on_source,
        predict_with_generate=self.args.predict_with_generate
        )
        if(os.path.exists(self.args.rawdata_save_dir)):
            self.dataset = load_from_disk(self.args.rawdata_save_dir)
        else:
            self.dataset = load_dataset(self.args.dataset_naqqme)
            self.dataset.save_to_disk(self.args.rawdata_save_dir)
            
    def prepare_data(self):
        """Download and tokenize or do preprocessing on complete dataset,
        because this is called on single gpu if your using mulitple gpu, 
        data here is not shared accross gpus."""


    def setup(self, stage: str):
        """splitting or transformations etc. 
        setup takes stage argument None by default or fit or test 
        for training and testing respectively.
        """
        Train_X, Valid_X, Train_Y, Valid_Y = train_test_split(
            self.dataset['train']['prompt'],self.dataset['train']['response'], 
            test_size = 0.1, random_state=self.args.random_state)
        Test_X, Test_Y = self.dataset['test']['prompt'], self.dataset['test']['response']
        self.train_dataset = BaseDataset(
            Train_X, Train_Y, self.tokenizer, **vars(self.args)
        )
        self.valid_dataset = BaseDataset(
            Valid_X, Valid_Y, self.tokenizer, **vars(self.args)
        )
        self.test_dataset = BaseDataset(
            Test_X, Test_Y, self.tokenizer, **vars(self.args)
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True, 
            collate_fn=self.data_collator, 
            num_workers=32)

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False, 
            collate_fn=self.data_collator, 
            num_workers=32)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False, 
            collate_fn=self.data_collator, 
            num_workers=32)