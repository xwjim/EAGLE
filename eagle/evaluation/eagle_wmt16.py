from datasets import load_dataset, load_metric, load_from_disk

from transformers import AutoTokenizer

from ..model.ea_model import EaModel
from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *
from ..model.choices import *
from transformers import LlamaForCausalLM as CausalLM

import time, torch, os

import argparse

from tqdm import tqdm


prompt = "Translate this German sentence to English. German: {de} English: "

prompt_chat = "Translate this German sentence to English and do not output anything more. German: {de} English: "

def evaluate(model, dataset, args):
    bleu = load_metric('bleu')
    predictions = []
    references = []
    # cnt = 0
    tokenizer = model.get_tokenizer()
    logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    total_time = 0 
    total_token = 0
    for entry in tqdm(dataset["train"]):
        entry = entry['translation']
        if args.model_type == "chat":
            input_str = prompt_chat.format(**entry)
        else:
            input_str = prompt.format(**entry)
        torch.cuda.empty_cache()
        input_ids = tokenizer([input_str]).input_ids
        output_ids, new_token, idx, coume_time = ea_forward(
            torch.as_tensor(input_ids).cuda(),
            model,
            tokenizer,
            eval(args.tree_choices),
            logits_processor,
        )
        total_time += coume_time
        total_token += new_token
        torch.cuda.synchronize()
        output_ids = output_ids[0][len(input_ids[0]):]
        # be consistent with the template's stop_token_ids
        output_str = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        predictions.append(output_str)
        references.append(entry['en'])
        # if cnt > 2:
        #     break
        # cnt += 1
    references = [[r.split()] for r in references]
    predictions = [p.split() for p in predictions]
    results = bleu.compute(predictions=predictions, references=references)
    pretty_results = results['bleu']
    return pretty_results, total_token/total_time



def ea_forward(input_ids, model, tokenizer, tree_choices, logits_processor=None, max_steps=512, max_new_token=128):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices, device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.base_model.lm_head.weight.device)
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
    )
    new_token = 0
    t0 = time.time()

    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["tree_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
            tree_candidates, tree_buffers["b_indices"],
        )
        input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits_processor,
            logits,
            tree_logits,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state,
            hidden_state_new,
            sample_p
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > max_new_token:
            break
        if input_ids.shape[1] > 1960:
            break
    return input_ids, new_token, idx, time.time() - t0


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="Data path of the dataset", default="/data4/wxu/github/Ouroboros/benchmark/DualAuthor/wmt16_small")
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-llama-2-chat-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument('--model_type', type=str, default='chat')
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    args = parser.parse_args()  
    return args


def main():
    args = parse()

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    if args.data_path:
        dataset = load_dataset(args.data_path)
    else:
        dataset = load_dataset('wmt16', 'de-en')

    result, latency = evaluate(model, dataset if args.data_path else dataset['test'], args)
    print(result)
    print(f"total speed: {latency:.2f} tok / s")
    return


if __name__ == "__main__":
    main()

'''
Yi:
    greedy: python test_wmt16.py --target_model <target_model_path> --data_path <data_path> 
    ouroboros: python test_wmt16.py --target_model <target_model_path> --data_path <data_path> --draft_model <draft_model_path> --ouroboros

Llama-70b:
    greedy: python test_wmt16.py --target_model <target_model_path> --data_path <data_path> --model_type chat
    ouroboros:   python test_wmt16.py --target_model <target_model_path> --data_path <data_path> --model_type chat --draft_model <draft_model_path> --ouroboros --window_size 13 --guess_set_size 13 --lookahead_level 5 --gamma 4 
'''