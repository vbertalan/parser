## Training SentenceTransformers model from scratch

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

path = "sample-logs.txt"


## Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

tokenizer.enable_truncation(max_length=2521)

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

tokenizer.save_model("NovoModelo")

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./NovoModelo", max_len=512)

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
    output_dir="./LogFiles",
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
trainer.save_model("./NovoModelo")