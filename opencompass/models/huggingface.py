import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import threading
import concurrent.futures
import signal


from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
from opencompass.utils.cal import api_map
from opencompass.utils.parser import extract_api

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

PromptType = Union[PromptList, str]

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
):
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

def calculate(cal_json, logger=None):
    # TODO: should fix in future, ugly writing style
    try:
        api_name = cal_json["ActionName"].replace(" ", "")
        api_args = cal_json["Args"]
        result = api_map[api_name](api_args)
        output = {"result": result}
    except Exception as e:
        output = {"result": ""}
        if logger is not None:
            logger.error(f"Error in calculate: {e},input: {cal_json}")
    return output

@MODELS.register_module()
class HuggingFace(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        from opencompass.utils.fileio import patch_hf_auto_model
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        patch_hf_auto_model(hf_cache_dir)
        self.logger = get_logger()
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        self.logits_processor = prepare_logits_processor(
            temperature=0.00000001, repetition_penalty=1.0, top_p=1.0, top_k=-1
        )
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        print('==tokenizer_path=new==',tokenizer_path)
        if tokenizer_kwargs:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path if tokenizer_path else path)
            
        if 'Qwen' in  path :
            print('==tokenizer_path=Qwen==',path)
            # self.tokenizer.bos_token = '<s>'
            # self.tokenizer.eos_token = '</s>'
            # self.tokenizer.eos_token_id = 151643
            self.tokenizer.pad_token_id = 151643
            # self.tokenizer.pad_token = self.tokenizer.eos_token
                    
        if self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer. '
                                'Using eos_token_id as pad_token_id.')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print('====00000==_load_model===Qwen=====',path)
        if 'Qwen' in  path :
            print('======_load_model===Qwen=====',path)
            self.model.config.eos_token_id = 151643
            self.model.config.pad_token_id = 151643


    
    def generate(self, inputs: List[str], max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs,
                                        max_out_len=max_out_len,
                                        **kwargs)
        else:
            return sum((self._single_generate(
                inputs=[input_], max_out_len=max_out_len, **kwargs)
                        for input_ in inputs), [])

    def _batch_generate(self, inputs: List[str], max_out_len: int,
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        print('_batch_generate')
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len -
                                                  max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds
    
    def _single_generate(self, inputs: List[str], max_out_len: int, count=0, 
                         **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # print("input: ", inputs)
        
        input_ids = self.tokenizer(inputs, truncation=True)["input_ids"]

        model_input_ids = torch.tensor(input_ids, device=self.model.device)
        # input_ids = torch.tensor(input_ids, device=self.model.device)
        input_ids = input_ids[0]
        output_ids = list(input_ids)
        device = self.model.device
        
        # To accommodate the PeftModel, parameters should be passed in
        # key-value format for generate.
        
        stop_token_ids = [self.tokenizer.eos_token_id]
        
        max_out_len = 2048 
        max_new_tokens = max_out_len - len(input_ids) - 8
        
        past_key_values = out = None
        gen_tokens = []

        if "</API>" in self.tokenizer.additional_special_tokens:
            api_token_index = self.tokenizer.additional_special_tokens.index("</API>")
            api_token_id = self.tokenizer.additional_special_tokens_ids[api_token_index]
            # all_api_tokens = self.tokenizer.additional_special_tokens
            # all_api_ids = self.tokenizer.additional_special_tokens_ids(all_api_tokens)
            api_start_index = self.tokenizer.additional_special_tokens.index("<API>")
            api_start_id = self.tokenizer.additional_special_tokens_ids[api_start_index]
        else:
            api_token_id = 100000000
        
        # print("api token id, ", api_token_id)

        for i in range(max_new_tokens):
            if i == 0 or restart_gen:
                out = self.model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
                restart_gen = False
            else:
                out = self.model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            
            tmp_output_ids = None
        
            last_token_logits = self.logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            
            # Greedy decoding
            token = int(torch.argmax(last_token_logits))
            
            gen_tokens.append(token)
            output_ids.append(token)
            try:
                if token == api_token_id : # </API>
                    print("Start detecting API")
                    msg = self.tokenizer.decode(gen_tokens).replace(" ", "")
                    api_json = extract_api(msg)
                    print(f"api_json is {api_json}")
                    if len(api_json) != 0:
                        api_json = api_json[-1]
                    if api_json is None:
                        print("No API detected!")
                        pass
                    if type(api_json) != dict:
                        print(f"The API Json is not in a valid format: {api_json}")
                        
                    elif "ActionName" not in api_json or "Args" not in api_json:
                        print(f"API detected, but the parameters are incomplete: {api_json}")
                    else:
                        print(f"Attempting to call API: {api_json['ActionName']}, parameter is: {api_json['Args']}")
                        # Define a function to handle signals
                        def handler(signum, frame):
                            raise Exception("Timeout!")
                        # Set up the signal handling function
                        signal.signal(signal.SIGALRM, handler)
                        try:
                            # Set the timer
                            signal.alarm(60)
                            try:
                                answer = calculate(api_json)
                            except MemoryError:
                                print("Unable to allocate so much memory, skipping")
                            # # Cancel the timer
                            # signal.alarm(0)
                        except Exception as e:
                            print("calculate function took too long to complete.")
                            answer = None
                            continue
                        finally:
                            signal.alarm(0)  # Turn off timeout
      
                        if answer is None or len(str(answer["result"])) > 2000:
                            print(f"Call {api_json['ActionName']} failed, parameter is {api_json['Args']}")
                        else:
                            answer = answer["result"]
                            print(f"{api_json['ActionName']} result is {answer}")
                            answer = f"=> {str(answer)}</thought>"
                            answer_tokens = self.tokenizer(
                                [answer],
                            ).input_ids[0][1:]
                            gen_tokens.extend(answer_tokens)
                            output_ids.extend(answer_tokens)
                            # Restart generation
                            input_ids += gen_tokens
                            gen_tokens = []
                            restart_gen = True
                            past_key_values = out = None
                            print("Restart generation")
            except ValueError:
                continue 

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:                    
                if i == max_new_tokens - 1:
                    print("Maximum length termination")
                else:
                    print("</s> termination")
                break
                

        
        if not self.extract_pred_after_decode:
            api_outputs = torch.as_tensor([output_ids], device=device)[:, model_input_ids.shape[1]:]
            
        api_decodeds = self.tokenizer.batch_decode(api_outputs, skip_special_tokens=False)
        
        if self.extract_pred_after_decode:

            api_decodeds = [token[len_:] for token, len_ in zip(api_decodeds, prompt_lens)]

        # print('api_decodeds--huggingface.py: ', api_decodeds)

        return api_decodeds

        
    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs[0], {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous()

        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().float().numpy() / lens
        return ce_loss

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args
        self._result = None

    def run(self):
        self._result = self._target(*self._args)

    @property
    def result(self):
        return self._result
@MODELS.register_module()
class HuggingFaceCausalLM(HuggingFace):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModelForCausalLM

        model_kwargs.setdefault('torch_dtype', torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        if 'Qwen' in  path :
            print('======_load_model===Qwen=====',path)
            self.model.config.pad_token_id = 151643
        self.model.eval()
