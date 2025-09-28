import os
import argparse
import torch
import random
from pathlib import Path
import pickle
from dataclasses import dataclass
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from accelerate import Accelerator
import json
import wandb
wandb.init(mode="offline")  # 离线模式初始化


accelerator = Accelerator()
IGNORE_INDEX = -100
MAX_LENGTH = 1024
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class CADDataset(Dataset):
    def __init__(self, pickle_fn, llama_tokenizer=None):
        if not os.path.exists(pickle_fn):
            raise ValueError(f"{pickle_fn} does not exist")
        self.inputs = pickle.load(open(pickle_fn, "rb"))
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self.inputs):
            raise ValueError(f"Index {index} out of range")
        val = self.inputs[index]
        val = self.tokenize(val)
        return val

    def tokenize(self, input_str):
        flag = 0
        while 1:
            local_loops_str = extract_loop_contents(input_str)
            random.shuffle(local_loops_str)
            for i in local_loops_str:
                if i in train_loop and input_str.count(i)==1:
                    prompt = (
                        'Below is a partial description of a CAD sequence where one '
                        'command has been replaced with the string "[loop mask]":\n'
                    )

                    prompt = prompt + input_str.replace(i, '[loop mask]') + "\n"
                    prompt += (
                        "Generate a string that could replace \"[loop mask]\" in the CAD sequence. The string denotes "
                    )
                    loop_caption = train_loop[i].replace('An ', 'an ').replace('A ', 'a ')
                    if 'unit' in loop_caption and 'with' in loop_caption and random.random()>0.5:
                        loop_caption = loop_caption.split(' with')[0]+'.'
                    prompt += loop_caption +'\n\n'
                    prompt += i
                    val_len = self.llama_tokenizer(
                        (prompt + self.llama_tokenizer.eos_token).split('\n\n')[-1],
                        max_length=MAX_LENGTH,
                        return_tensors="pt",
                        truncation=True,
                    ).input_ids.shape[1] - 1
                    tokens = self.llama_tokenizer(
                        prompt + self.llama_tokenizer.eos_token,
                        max_length=MAX_LENGTH,
                        return_tensors="pt",
                        truncation=True,
                    )
                    flag += 1
                    break
            if flag==0:
                input_str = random.choice(self.inputs)
            else:
                break
        input_ids = labels = tokens.input_ids[0]
        input_id_lens = label_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            input_id_lens=input_id_lens,
            labels=labels,
            label_lens=label_lens,
            val_len=val_len,
        )




def count_curve(loop):
    c_line = find_all_curve_end_positions(loop, "line")
    c_arc = find_all_curve_end_positions(loop, "arc")
    c_circle = find_all_curve_end_positions(loop, 'circle')
    dict_curve_type = {}
    for c_i in c_line:
        dict_curve_type[c_i] = '[line mask] '
    for c_i in c_arc:
        dict_curve_type[c_i] = '[arc mask] '
    for c_i in c_circle:
        dict_curve_type[c_i] = '[circle mask] '
    list_curve = ''
    c_all = c_line + c_arc + c_circle
    c_all.sort()
    for c_i in c_all:
        list_curve += (dict_curve_type[c_i])
    return list_curve

def find_all_curve_end_positions(string,curve_end):
    positions = []
    start = 0
    while True:
        position = string.find(curve_end, start)
        if position == -1:
            break
        positions.append(position)
        start = position + len(curve_end)
    return positions

def interleave_lists_with_zeros(list1, list2):
    result = [0]
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        result.append(list1[i])
        result.append(list2[i])
    result.extend(list1[min_length:])
    result.extend(list2[min_length:])
    return result

def replace_at_index(original_string, index, index_h, replacement):
    if index < 0 or index >= len(original_string):
        return "Index out of range"
    return original_string[:index] + replacement + original_string[index_h:]


def extract_loop_contents(s):
    elements = s.split()
    loop_end_indices = [i for i, elem in enumerate(elements) if elem == '<loop_end>']
    results = []

    for i in loop_end_indices:
        j = i - 1
        start = -1
        while j >= 0:
            elem = elements[j]
            if elem.startswith('<') and elem.endswith('>'):
                if elem == '<curve_end>':
                    j -= 1
                else:
                    start = j
                    break
            else:
                j -= 1
        if start != -1:
            content = elements[start + 1:i]
        else:
            content = elements[:i]
        results.append(' '.join(content)+' <loop_end>')
    return results

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances]
            for key in ("input_ids", "labels")
        )
        val_len = [instance['val_len'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        for i in range(len(labels)):

            cur_len= labels[i].shape[0]
            if cur_len < MAX_LENGTH:
                labels[i][:cur_len - val_len[i]] = IGNORE_INDEX
            else:
                labels[i][:] = IGNORE_INDEX

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_datasets(args, llama_tokenizer):
    datasets = {
        "train": CADDataset(
            str(args.data_path / "train.pkl"),
            llama_tokenizer=llama_tokenizer,
        ),
        "val": CADDataset(
            str(args.data_path / "test.pkl"),
            llama_tokenizer=llama_tokenizer,
        ),
    }

    return datasets


def setup_training_args(args):
    output_dir = args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["cad_ids"],  # this is just to get trainer to behave how I want
    )
    return training_args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def setup_model(args, rank):
    if args.model_name=="8B":
        model_id = 'meta-llama/Meta-Llama-3-8B' # path-to-LLM-checkpoints

        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map={"": rank})
        llama_tokenizer = pipeline.tokenizer
        model = pipeline.model
    elif args.model_name=="70B":
        model_id = "meta-llama/Meta-Llama-3-70B"
        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model="meta-llama/Meta-Llama-3-70B", load_in_8bit=True, model_kwargs={"torch_dtype": torch.bfloat16},
                                         device_map={"": rank})
        llama_tokenizer = pipeline.tokenizer
        model = pipeline.model
    else:
        llama_options = args.model_name.split("-")
        is_chat = len(llama_options) == 2
        model_size = llama_options[0]

        def llama2_model_string(model_size, chat):
            chat = "chat-" if chat else ""
            return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

        model_string = llama2_model_string(model_size, is_chat)
        print(f"Model size: {model_string}")
        model = LlamaForCausalLM.from_pretrained(
            model_string,
            load_in_8bit=args.fp8,
            device_map={"": rank},
        )

        llama_tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )


    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer

def setup_trainer(args):
    training_args = setup_training_args(args)
    model, llama_tokenizer = setup_model(args, training_args.local_rank)
    datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=data_collator,
    )

    return trainer


def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="exp")
    parser.add_argument("--model-name", default="8B")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/basic")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=500, type=int)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "CADLLM"
    print(args)
    with open(str(args.data_path / "train_loop.json"), 'r') as f:
        train_loop = json.load(f)
    main(args)
