## Training SentenceTransformers model from scratch

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import os

os.chdir('/home/vberta/projects/def-aloise/vberta/Parser-CC/parser')
path = "Ciena-mini.txt"
length = 668

## Check if there is a GPU
#!nvidia-smi

## Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.enable_truncation(max_length=length)

## Customize training
tokenizer.train(files=[path], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<cls>",
    "<bos>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("Ciena-Mini-Tokenizer")

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=length,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./Ciena-Mini-Tokenizer", max_len=length)

model = RobertaForMaskedLM(config=config)
parametros = model.num_parameters()
print("O numero de parametros e {}".format(parametros))

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=path,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./Ciena-Mini-Transformer",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("./Ciena-Mini-Transformer")