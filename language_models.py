import os
import time
import torch
import gc
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import threading

import openai
import anthropic
import google.generativeai as palm
import requests
import json

import config
from conv_builder import ConvBuilder
from common import ConversationTemplateUtil

# https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/language_models.py

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """Generate batch responses"""
        raise NotImplementedError

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float):
        """Generate batch responses by threads"""
        raise NotImplementedError

# API base class for models that use an API
class BaseAPIModel(LanguageModel):
    API_RETRY_SLEEP = 15
    API_ERROR_OUTPUT = "$ERROR$" 
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 120

    def __init__(self, model_name):
        super().__init__(model_name)

class LocalModel(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        super().__init__(model_name)
        self.model = model
        self.tokenizer = tokenizer

# HuggingFace
class HuggingFace(LocalModel):
    def __init__(self, model_name, model, tokenizer):
        super().__init__(model_name, model, tokenizer)
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self,
                         full_prompts_list,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        inputs = self._prepare_inputs(full_prompts_list)
        output_ids = self._generate_outputs(inputs, max_n_tokens, temperature, top_p)
        outputs_list = self._process_outputs(output_ids, inputs)
        self._cleanup(inputs, output_ids)
        return outputs_list

    def _prepare_inputs(self, full_prompts_list):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        return {k: v.to(self.model.device.index) for k, v in inputs.items()}

    def _generate_outputs(self, inputs, max_n_tokens, temperature, top_p):
        generation_config = {
            "max_new_tokens": max_n_tokens,
            "eos_token_id": self.eos_token_ids,
            "top_p": top_p
        }
        
        if temperature > 0:
            generation_config.update({
                "do_sample": True,
                "temperature": temperature
            })
        else:
            generation_config.update({
                "do_sample": False,
                "temperature": 1
            })
            
        return self.model.generate(**inputs, **generation_config)

    def _process_outputs(self, output_ids, inputs):
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def _cleanup(self, inputs, output_ids):
        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

    def extend_eos_tokens(self):
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675])

# GPT API
class GPT(BaseAPIModel):
    def __init__(self, model_name, api_key=config.OPENAI_API_KEY):
        super().__init__(model_name)
        self._setup_api(api_key)

    def _setup_api(self, api_key):
        self.api_key = api_key
        self.base_url = config.OPENAI_API_BASE

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self._make_api_call(conv, max_n_tokens, temperature)
                output = self._parse_response(response)
                break
            except Exception as e:
                print("OpenAI Generate Error: ", type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def _make_api_call(self, conv, max_n_tokens, temperature):
        payload = {
            "model": self.model_name,
            "messages": conv,
            "temperature": temperature,
            "max_tokens": max_n_tokens
        }
        headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return requests.post(self.base_url, headers=headers, json=payload)

    def _parse_response(self, response):
        try:
            return response.json()["choices"][0]["message"]["content"]
        except:
            print(response.text)
            return self.API_ERROR_OUTPUT

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]], 
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self.generate, convs_list,
                                   [max_n_tokens]*len(convs_list),
                                   [temperature]*len(convs_list),
                                   [top_p]*len(convs_list))
        return list(results)
    

# One API
class OneAPI(BaseAPIModel):
    def __init__(self, model_name, api_key=config.ONE_API_KEY):
        super().__init__(model_name)
        self._setup_api(api_key)

    def _setup_api(self, api_key):
        self.api_key = api_key
        self.base_url = config.ONE_API_BASE

    def generate(self, conv: List, max_n_tokens: int, temperature: float, top_p: float):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self._make_api_call(conv, max_n_tokens, temperature, top_p)
                output = self._parse_response(response)
                break
            except Exception as e:
                print(f"OneAPI Generate Error: {type(e)} {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output

    def _make_api_call(self, conv, max_n_tokens, temperature, top_p):
        payload = {
            "messages": conv,
            "model": self.model_name,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        return requests.post(self.base_url, headers=headers, json=payload)

    def _parse_response(self, response):
        try:
            return response.json()["choices"][0]["message"]["content"]
        except:
            print(response.text)
            return self.API_ERROR_OUTPUT

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list], []*len(convs_list)

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self.generate, convs_list,
                                   [max_n_tokens]*len(convs_list),
                                   [temperature]*len(convs_list),
                                   [top_p]*len(convs_list))
        return list(results)

if __name__ == "__main__":
    models = ["gpt-4o-mini"]

    for model in models:
        print("======="*10)
        print("Model: {}".format(model))

        lm = OneAPI(model)
        conv = ConversationTemplateUtil().get_template(model)

        conv.append_message(conv.roles[0], "Please tell me a joke.")
        
        prompt_list = [conv.to_openai_api_messages()]
        print(prompt_list)

        print(lm.batched_generate(prompt_list, 100, 1.0, 1.0))
