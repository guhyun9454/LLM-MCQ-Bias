
import os
import sys
import json
import logging
from eval_clm_utils import (
    parse_arguments,
    prepare_eval,
)
from utils import (
    _orange, _blue, _purple,
    eval_all_samples,
    get_accuracy,
    get_bootstrap_accuracy_std,
    save_results,
    patch_open,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import gc

import pynvml
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


def logging_cuda_memory_usage():
    logger.info("******** Memory usage ********")
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))


def main():
    patch_open()

    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()
    if len(args.eval_names) == 0:
        exit()

    os.makedirs('models', exist_ok=True)

    try:
        toker = AutoTokenizer.from_pretrained(
            args.pretrained_model_path,
            use_fast=True,
            add_bos_token=False, add_eos_token=False,
            trust_remote_code=True,
            cache_dir='models',
        )
        logger.info("Tokenizer loaded with use_fast=True")
    except Exception as e_fast:
        logger.warning(f"Failed to load tokenizer with use_fast=True: {e_fast}. Retrying with use_fast=False.")
        try:
            toker = AutoTokenizer.from_pretrained(
                args.pretrained_model_path,
                use_fast=False,
                add_bos_token=False, add_eos_token=False,
                trust_remote_code=True,
                cache_dir='models',
            )
            logger.info("Tokenizer loaded with use_fast=False")
        except Exception as e_slow:
            logger.exception(
                f"Failed to load tokenizer (use_fast=True/False) for {args.pretrained_model_path}: {e_slow}"
            )
            return

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            device_map='auto',
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            cache_dir='models',
        )
    except Exception as e_model:
        logger.exception(f"Failed to load model for {args.pretrained_model_path}: {e_model}")
        return
    logging_cuda_memory_usage()

    printed_verbose_prompt = False

    for eval_name in args.eval_names[::1]:
        (
            subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn
        ) = prepare_eval(args, eval_name)
        for subject in subjects[::1]:
            if os.path.exists(f'{args.save_path}/{subject}.jsonl'):
                logger.info(f"Results already exist: {args.save_path}/{subject}.jsonl")
                continue

            logger.info(_blue(f"Preparing: {subject}"))
            few_shot_samples = prepare_few_shot_samples(subject)
            eval_samples = prepare_eval_samples(subject)
            eval_fn = prepare_eval_fn(model, toker, few_shot_samples)

            if getattr(args, 'verbose', False) and not printed_verbose_prompt and len(eval_samples) > 0:
                try:
                    # Build exactly the same prompt string fed into the model (for first sample only)
                    if args.setting in ['perm', 'cyclic']:
                        probing_inputs, _, _ = eval_samples[0]
                        sys_msg, eval_sample = probing_inputs[0]
                        input_text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
                                input_text += s + '\n\n'
                        input_text += eval_sample
                        # Match eval_fn_perm behavior for BPE space prefix
                        try:
                            id_space = toker.encode(': A', add_special_tokens=False)[-1]
                            id_nospace = toker.encode(':A', add_special_tokens=False)[-1]
                            bpe_has_space_prefix = (id_space != id_nospace)
                        except Exception:
                            bpe_has_space_prefix = True
                        if not bpe_has_space_prefix:
                            input_text += ' '
                    elif args.setting in ['noid']:
                        inputs, _, _ = eval_samples[0]
                        sys_msg, eval_sample = inputs
                        input_text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
                                input_text += s + '\n\n'
                        input_text += eval_sample
                        # no extra space in noid eval
                    else:
                        inputs, _, _ = eval_samples[0]
                        sys_msg, eval_sample = inputs
                        input_text = sys_msg + '\n\n'
                        if args.num_few_shot > 0:
                            for s in few_shot_samples[:args.num_few_shot]:
                                input_text += s + '\n\n'
                        input_text += eval_sample
                        # Match eval_fn_base behavior for BPE space prefix
                        try:
                            id_space = toker.encode(': A', add_special_tokens=False)[-1]
                            id_nospace = toker.encode(':A', add_special_tokens=False)[-1]
                            bpe_has_space_prefix = (id_space != id_nospace)
                        except Exception:
                            bpe_has_space_prefix = True
                        if not bpe_has_space_prefix:
                            input_text += ' '

                    logger.info(_purple("====== VERBOSE: Example prompt fed to model (first sample) ======"))
                    logger.info(input_text)
                    logger.info(_purple("========================= END OF PROMPT ========================="))
                    printed_verbose_prompt = True
                except Exception as e:
                    logger.warning(f"Failed to build verbose prompt example: {e}")

            logger.info(_blue(f"Run started: {subject}"))
            results = eval_all_samples(
                eval_fn, eval_samples,
                name=f'{args.task},{args.num_few_shot},{args.setting},{subject}',
                threads=torch.cuda.device_count() if 'falcon' not in args.pretrained_model_path else 1,
            )
            gc.collect()
            torch.cuda.empty_cache()

            metrics = None
            if args.setting not in ['perm', 'cyclic'] and len(results) > 0:
                metrics = {'type': 'metric', 'data': {}}
                metrics['data']['accuracy'] = get_accuracy(results)
                metrics['data']['boostrap_std'] = get_bootstrap_accuracy_std(results)
                logger.info("Final report:")
                for key, value in metrics['data'].items():
                    logger.info(f"{key}: {value}")
            logger.info(_orange(f"Run completed: {subject}"))

            save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
            logger.info(f"Results saved: {subject}")

            logging_cuda_memory_usage()


if __name__ == "__main__":
    main()

