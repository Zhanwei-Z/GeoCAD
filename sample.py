import random
import argparse
import json
import transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import PeftModel
from finetune import MAX_LENGTH
import pickle
import os
import wandb
wandb.init(mode="offline")  # 离线模式初始化

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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

loop_caption_list = ['an irregular quadrilateral.','a kite quadrilateral.',
                     'an ordinary trapezoid.','a right trapezoid.',
                     'an isosceles trapezoid.','a parallelogram.',
                     'a rhombus.','a rectangle.','a square.',
                     'an isosceles right triangle.','an isosceles acute triangle.','an isosceles obtuse triangle.',
                     'a right triangle.','an acute triangle.','an obtuse triangle.',
                     'a shape that looks like a rectangle with one corner cut off.',
                     'a shape that looks like a rectangle with a smaller rectangle cut off.',
                     'a quarter-circle.','a three-quarter circle.','a semicircle.', 'a circle.',
                     'a minor-arc sector of a circle.','a major-arc sector of a circle.',
                     'a minor-arc loop resembling a bow.','a major-arc loop.']

def convert(input_str,loop_caption):
    local_loops_str = extract_loop_contents(input_str)
    local_loops_str = local_loops_str[0:1]  # Take the first loop as an example
    list_p = []
    count_loop = {}
    for i in local_loops_str:
        if i not in count_loop.keys():
            count_loop[i] = 0
        else:
            count_loop[i] += 1
        if count_loop[i] == 0:
            cur_str = input_str.replace(i, '[loop mask]', 1)
        else:
            cur_str = input_str.replace(i, '[loop mask]', count_loop[i] + 1)
            cur_str = cur_str.replace('[loop mask]', i, count_loop[i])

        prompt = (
            'Below is a partial description of a CAD sequence where one '
            'command has been replaced with the string "[loop mask]":\n'
        )

        prompt = prompt + cur_str + "\n"
        prompt += (
            "Generate a string that could replace \"[loop mask]\" in the CAD sequence. The string denotes "
        )
        prompt += loop_caption + '\n\n'
        list_p.append(prompt)
    return list_p





def prepare_model_and_tokenizer(args):
    if args.model_name=="8B":
        model_id = 'meta-llama/Meta-Llama-3-8B' # path-to-LLM-checkpoints
        print(f"Model size: {model_id}")
        pipeline = transformers.pipeline("text2text-generation",
                                         model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map='auto')
        tokenizer = pipeline.tokenizer
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
            load_in_8bit=True,
            device_map="auto",
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            model_string,
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
        )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")

    return model, tokenizer


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


def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    prompts = []
    originals = []
    for _ in range(args.num_samples):
        with open('./cad_data/processed_data/test.pkl', "rb") as f:
            data_test = pickle.load(f)
        input_str = data_test[5613]
        for k in loop_caption_list:
            loop_caption = k
            prompt = convert(input_str, loop_caption)
            prompts += prompt
            originals += [loop_caption]*len(prompt)
    outputs = []
    delimiter = "\n\n"
    # delimiter = 'in the CAD sequence:'
    while len(outputs) < len(prompts):
        batch_prompts = prompts[len(outputs): len(outputs) + args.batch_size]

        batch = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=MAX_LENGTH,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for i in range(len(gen_strs)):
            gen_str = gen_strs[i].split(delimiter, 2)[1] + ' '
            # print(gen_str)
            loop_number = gen_str.count('loop_end')
            prompt = batch_prompts[i]
            prompt = prompt.split('\n')[1]
            # print(prompt)
            multi_loop = '[loop mask] '*loop_number
            prompt = prompt.replace(multi_loop, gen_str).replace('\n', '')
            print(gen_strs[i])
            outputs.append(prompt)

        print(f"Generated {len(outputs)}/{len(prompts)}samples.")
        with open(os.path.dirname(args.model_path)+'conditional_samples_'+str(args.num_samples)+'_'+args.mask_type+'_mask.json', "w") as f:
            for prompt, output, original in zip(prompts, outputs, originals):
                f.write(json.dumps({"prompt": prompt, "output": output, "original":original}) + "\n")
    with open(os.path.dirname(args.model_path)+'conditional_samples_'+str(args.num_samples)+'_'+args.mask_type+'_mask.json', "w") as f:
        for prompt, output, original in zip(prompts, outputs, originals):
            f.write(json.dumps({"prompt": prompt, "output": output, "original":original}) + "\n")
    print(len(prompts))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="8B")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="_cad_samples.json")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mask_type", type=str, default="loop")
    parser.add_argument("--use_fixed_demo", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    conditional_sample(args)
