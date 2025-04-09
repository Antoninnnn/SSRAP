The repository for ECEN766 course work


git clone 2 repository

```shell
git clone https://github.com/ai4protein/VenusPLM.git

git clone https://github.com/ai4protein/ProSST.git

```

For yining's environment(cuda127):

```shell
conda create -n prosst
conda activate prosst
```

```shell
cd ProSST

## I have met some problem for certain packages in the requirements.txt, I skipped some of them and pip install individually.
pip install -r requirements.txt

# install some pip packages individually
pip install torch-scatter torch-sparse torch-spline-conv  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html


pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```


For alphabet tokens:
ProSST Tokenizer

```


EsmTokenizer(name_or_path='AI4Protein/ProSST-2048', vocab_size=25, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'cls_token': '<cls>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
	0: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<cls>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("<eos>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	23: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	24: AddedToken("<mask>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}

```
esm_msa tokenizer
```

['<cls>',
 '<pad>',
 '<eos>',
 '<unk>',
 'L',
 'A',
 'G',
 'V',
 'S',
 'E',
 'R',
 'T',
 'I',
 'D',
 'P',
 'K',
 'Q',
 'N',
 'F',
 'Y',
 'M',
 'H',
 'W',
 'C',
 'X',
 'B',
 'U',
 'Z',
 'O',
 '.',
 '-',
 '<null_1>',
 '<mask>']

```