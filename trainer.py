import os

import sys
import json

from datasets import arrow_dataset
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

from formula_datasets import LoadedDataset



class SimpleLlamaTrainConfig:
    def __init__(self, **kwargs):
        self.lr = kwargs.pop("lr", 2e-4)
        self.dataset_load_amount = kwargs.pop("dataset_load_amount", 100_000)
        self.epochs = kwargs.pop("epochs", 3)
        self.save_dir = kwargs.pop("save_dir", None)
        self.train_dataset = kwargs.pop("train_dataset", None)
        self.eval_dataset = kwargs.pop("eval_dataset", None)
        self.eval_steps = kwargs.pop("eval_steps", 5000)
        self.save_steps = kwargs.pop("save_steps", 10_000)
        self.model_src = kwargs.pop("model_src", None)
        self.checkpoint_folder = kwargs.pop("checkpoint_folder", None)
        self.tokenizer_src = kwargs.pop("tokenizer_src", None)
        self.logging_steps = kwargs.pop("logging_steps", 200)
        self.max_seq_length = kwargs.pop("max_seq_length", 256)

        self.lora_r = kwargs.pop("lora_r", 16)
        self.use_flash_attention = kwargs.pop("use_flash_attention", False)
        self.per_device_train_batch_size = kwargs.pop("per_device_train_batch_size", None)
        self.eval_examples = kwargs.pop("eval_examples", 0)

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def format_instruction(sample):
    return sample["example"]

def eval_model(model:LlamaForCausalLM, ds:LoadedDataset, tok:PreTrainedTokenizer, conf:SimpleLlamaTrainConfig):
    losses = []
    for i_ex, (func, integ) in enumerate(ds):
        tokens = [tok.bos_token_id] + tok.encode(f"[INST] Integrate {func} dx [/INST]", add_special_tokens=False)
        L = len(tokens)
        tokens += tok.encode(integ, add_special_tokens=False) + [tok.eos_token_id]
        input_tokens = torch.LongTensor([tokens]).to("cuda")
        labels = input_tokens.detach().clone()
        labels[0, :L] = torch.full((1,L,), -100)
        forw_res = model.forward(input_ids = input_tokens, labels=labels, return_dict = True)
        gen = model.generate(input_tokens[:,:L], max_new_tokens=conf.max_seq_length, attention_mask=torch.full((1,L,), 1).to("cuda"), pad_token_id = tok.pad_token_id)
        print(f"example {i_ex}: {func} dx ===> {integ}")
        print(tok.decode(gen[0]))
        
        losses.append(forw_res["loss"].item())
    if len(losses) > 0:
        print(f"avg loss: {sum(losses)/len(losses)}")


def train(**kwargs):
    conf = SimpleLlamaTrainConfig(**kwargs)
    print(conf)

    if not conf.model_src or not conf.tokenizer_src:
        print("source not defined")
        return
    if conf.checkpoint_folder == "":
        print("checkpoint_folder can't be empty string")
        return
    if not conf.train_dataset:
        print("train dataset not defined")
        return
    if not conf.save_dir:
        print("checkpoint destination not defined")
        return

    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer_src, legacy=True, add_eos_token=True)
    tokenizer.pad_token_id=0
    tokenizer.padding_side = "right"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        conf.model_src,
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=conf.use_flash_attention,
        device_map="auto",
    )
    print("loaded model")
    model.model.embed_tokens.padding_idx = 0
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token

    model.config.pretraining_tp = 1
    print("number of parameters:", format(sum(p.numel() for p in model.parameters()), "_d"))

    ds = LoadedDataset(conf.train_dataset, conf.dataset_load_amount)
    print(len(ds), "total examples")

    ds = arrow_dataset.Dataset.from_list([{"example": f"[INST] Integrate {x[0]} dx [/INST] {x[1]}"} for x in ds])
    ev_ds = None
    if conf.eval_dataset not in [None, ""]:
        ev_ds = LoadedDataset(conf.eval_dataset, 300)
        ev_ds = arrow_dataset.Dataset.from_list([{"example": f"[INST] Integrate {x[0]} dx [/INST] {x[1]}"} for x in ev_ds])

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=conf.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)

    if conf.checkpoint_folder not in [None, ""]:
        print("loading PeftModel from", conf.checkpoint_folder)
        model = PeftModel.from_pretrained(model, conf.checkpoint_folder, is_trainable=True)
    else:
        print("checkpoint_folder not set; initializing random PEFT adapters")
        model = get_peft_model(model, peft_config)

    bs = conf.per_device_train_batch_size
    if bs == None:
        bs = 6 if conf.use_flash_attention else 4,

    args = TrainingArguments(
        output_dir=conf.save_dir,
        num_train_epochs=conf.epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=conf.logging_steps,
        save_strategy="steps",
        save_steps=conf.save_steps,
        evaluation_strategy="steps",
        eval_steps=conf.eval_steps,
        learning_rate=conf.lr,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=True,
        report_to="none"
    )

    trainer_callbacks = [PeftSavingCallback]
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        eval_dataset=ev_ds,
        peft_config=peft_config,
        max_seq_length=conf.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=format_instruction,
        args=args,
        callbacks=trainer_callbacks
    )

    # train
    trainer.train()

    # save model
    trainer.save_model()
    
    # evaluate performance
    eval(model.base_model,
        LoadedDataset(conf.train_dataset, conf.eval_examples), ev_ds,
        tokenizer, conf)


if __name__ == "__main__":
    parameters = {}
    for word_ind, word in enumerate(sys.argv):
        if word.startswith("--") and len(word) > 2 and word_ind < len(sys.argv)-1:
            parameters[word[2:]] = sys.argv[word_ind+1]
