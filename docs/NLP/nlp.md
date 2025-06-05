## TEXT Related NLP cleaning & feature engineering

```
!pip install --user -U nltk
```

```
import numpy as np
import pandas as pd
import nltk
nltk.download('all')
```

## Word Tokenizing

```
import nltk
from nltk.tokenize import word_tokenize as wt

sentence = '''This is going to be a great class don't you think'''
tokens = wt(sentence)
print(tokens)
```

## Sentence Tokenizing

```
import nltk
from nltk.tokenize import sent_tokenize as st

sentence = '''This is going to be a great class don't you think'''
tokens = st(sentence)
print(tokens)
```

## Char Tokenizer

## Custom Char Tokenizer with NLTK Style

```
from nltk.tokenize import RegexpTokenizer
sentence = '''This is going to be a great class don't you think'''
char_tokenizer = RegexpTokenizer(r'\w|[^\w\s]|\s')  # splits every char
tokens = char_tokenizer.tokenize(sentence)
print(tokens)
```

## ✅ 1. Hugging Face Tokenizers / Transformers

```
from tokenizers import Tokenizer, models, pre_tokenizers

# Initialize a character-level tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit('')  # splits every char
```

## ✅ 2. Keras / TensorFlow

The Tokenizer from keras.preprocessing.text can work at the character level.

```
from keras.preprocessing.text import Tokenizer

text = ["This is a test"]
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)
tokens = tokenizer.texts_to_sequences(text)
print(tokens)
```

## ✅ 3. TorchText (PyTorch)

You can define a custom tokenizer function and use it in a Field.

```
from torchtext.data import Field

char_tokenizer = lambda x: list(x)
TEXT = Field(tokenize=char_tokenizer)
```

## ✅ 4. SentencePiece (for training tokenizers)

You can train a model using ```--character_coverage=1.0``` and a suitable model type like BPE or char.

## ✅ 5. spaCy – Doesn't natively provide character-level tokenization, but you can override it.

```
import spacy

text = "Hello!"
tokens = list(text)
print(tokens)
```

## ✅ 6. Basic Python

```
text = "This is a test"
tokens = list(text)  # character-level tokenization
```

## Summary Table

| Library         | Char-level Support | Notes                                    |
| --------------- | ------------------ | ---------------------------------------- |
| Hugging Face    | ✅                  | With `tokenizers` or model config        |
| Keras / TF      | ✅                  | `Tokenizer(char_level=True)`             |
| TorchText       | ✅                  | Custom tokenizer with `list(x)`          |
| SentencePiece   | ✅                  | Can train char-level tokenizer           |
| spaCy           | 🚫 (manual only)   | Use `list(text)` for char-level manually |
| NLTK            | 🚫                 | No built-in char tokenizer               |
| Python (native) | ✅                  | `list(text)` is enough                   |



