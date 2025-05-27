import common
from language_models import GPT, HuggingFace, OneAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MODEL_CONFIGS
from collections import defaultdict
import json

# https://raw.githubusercontent.com/patrickrchao/JailbreakingLLMs/main/conversers.py

def load_attack_and_target_models(args):
    """Load attack and target language models"""
    attackLM = AttackLM(
        model_name=args.attack_model,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,
        top_p=ATTACK_TOP_P
    )

    # If the attack and target models are the same, reuse the loaded model
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Attack and target models are the same, reuse the loaded model.")
        preloaded_model = attackLM.model

    # Load target model
    targetLM = TargetLM(
        model_name=args.target_model,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,
        top_p=TARGET_TOP_P,
        preloaded_model=preloaded_model
    )

    return attackLM, targetLM

class ModelLoader:
    """Model Loader"""
    @staticmethod
    def load_model(model_name, device=None):
        model_path, template = ModelLoader.get_model_config(model_name)
        
        if "gpt" in model_name and "oneapi" not in model_name:
            print(f"Loading GPT model: {model_name}")
            return GPT(model_name), template
        elif "oneapi" in model_name:
            print(f"Loading OneAPI model: {model_name}/{model_path}")
            return OneAPI(model_path), template
        else:
            print(f"Loading HuggingFace model: {model_name}")
            return ModelLoader._load_huggingface_model(model_path, template)

    @staticmethod
    def _load_huggingface_model(model_path, template):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )

        ModelLoader._setup_tokenizer(tokenizer, model_path)
        
        return HuggingFace(model_path, model, tokenizer), template

    @staticmethod 
    def _setup_tokenizer(tokenizer, model_path):
        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def get_model_config(model_name):
        """Obtain model configuration"""
        model_configs = MODEL_CONFIGS
        if model_name not in model_configs:
            raise ValueError(f"Unknown model name: {model_name}")
            
        return model_configs[model_name]

class BaseLanguageModel:
    """Base class for language models"""
    def __init__(self, model_name, max_n_tokens, temperature, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.model, self.template = ModelLoader.load_model(model_name)

class AttackLM(BaseLanguageModel):
    """Attack language model"""
    def __init__(self, model_name, max_n_tokens, max_n_attack_attempts, temperature, top_p):
        super().__init__(model_name, max_n_tokens, temperature, top_p)
        self.max_n_attack_attempts = max_n_attack_attempts
        
        if ("vicuna" in model_name or "llama" in model_name) and "api" not in model_name and "groq" not in model_name:
            self.model.extend_eos_tokens()

    def _prepare_prompts(self, convs_list, prompts_list):
        """Prepare the prompts for model generation"""
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv = conv.copy()
            conv.append_message(conv.roles[0], prompt)
            
            if "gpt" in self.model_name or "oneapi" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "text-davinci" in self.model_name or "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "chatglm" in self.model_name:
                full_prompts.append(conv)
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
                
        return full_prompts

    def extract_json_for_mr_init_chain(self, text):
        """Extract JSON for multi-round initial chain from text"""
        return common.JsonProcessor.extract_mr_init_chain(text)

    def get_attack_mr_init_chain(self, convs_list, prompt_list):
        """Generate multi-round initial attack chains"""
        assert len(convs_list) == len(prompt_list), "Dialogue and prompt numbers do not match"

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        full_prompts = self._prepare_prompts(convs_list, prompt_list)

        for attempt in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
            
            outputs_list = self.model.batched_generate(
                full_prompts_subset,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )

            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if not full_output:
                    new_indices_to_regenerate.append(orig_index)
                    continue

                mr_conv, evaluation, _ = self.extract_json_for_mr_init_chain(full_output)

                if mr_conv is not None and evaluation is not None:
                    valid_outputs[orig_index] = {
                        "mr_conv": mr_conv,
                        "evaluation": evaluation
                    }
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if len(valid_outputs) > 0:
                break

        if any(output is None for output in valid_outputs):
            print(f"Generation failed after {self.max_n_attack_attempts} attempts")

        return valid_outputs

    def get_attack(self, convs_list, prompts_list):
        """Generate attack prompts with semantic similarity guidance"""
        assert len(convs_list) == len(prompts_list), "Dialogue and prompt numbers do not match"

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        candidate_outputs = [None] * batchsize

        full_prompts = self._prepare_prompts(convs_list, prompts_list)

        for attempt in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
            
            outputs_list = self.model.batched_generate(
                full_prompts_subset,
                max_n_tokens=self.max_n_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )

            if outputs_list is None:
                print("Error generating output")
                indices_to_regenerate = [indices_to_regenerate[0]]
                continue

            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if not full_output:
                    new_indices_to_regenerate.append(orig_index)
                    continue

                attack_dict, json_str = common.JsonProcessor().extract_single_round(full_output)

                if attack_dict is not None and attack_dict["prompt"] not in ["[new prompt]", "[新提示]"]:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break

        # Generate candidate outputs for failed attempts
        for i in range(len(valid_outputs)):
            if valid_outputs[i] is None:
                valid_outputs[i] = candidate_outputs[i]

        if any(output is None for output in valid_outputs):
            print(f"Generation failed after {self.max_n_attack_attempts} attempts")

        return [output for output in valid_outputs if output is not None]

class TargetLM(BaseLanguageModel):
    """Target language model"""
    def __init__(self, model_name, max_n_tokens, temperature, top_p, preloaded_model=None):
        if preloaded_model is None:
            super().__init__(model_name, max_n_tokens, temperature, top_p)
        else:
            self.model_name = model_name
            self.temperature = temperature
            self.max_n_tokens = max_n_tokens
            self.top_p = top_p
            self.model = preloaded_model
            _, self.template = ModelLoader.get_model_config(model_name)

    def get_response(self, convs_list):
            full_prompts = convs_list
            retry_attempts = 5

            indices_to_regenerate = list(range(len(full_prompts)))
            valid_outputs = [None] * len(full_prompts)
            valid_attentions = [None] * len(full_prompts)

            for attamp in range(retry_attempts):

                full_prompts_subset = [full_prompts[i]
                                        for i in indices_to_regenerate]
                
                outputs_list = self.model.batched_generate_by_thread(full_prompts_subset,
                                                                max_n_tokens=self.max_n_tokens,
                                                                    temperature=self.temperature,
                                                                    top_p=self.top_p)
                
                if outputs_list is None:
                    print("Error in generating output.")
                    indices_to_regenerate = [indices_to_regenerate[0]]
                    continue

                # Check for valid outputs and update the list

                new_indices_to_regenerate = []

                for i, full_output in enumerate(outputs_list):
                    orig_index = indices_to_regenerate[i]

                    if full_output is not None:
                        # Update the conversation with valid generation
                        valid_outputs[orig_index] = full_output
                    else:
                        new_indices_to_regenerate.append(orig_index)

                # Update indices to regenerate for the next iteration
                indices_to_regenerate = new_indices_to_regenerate

                # If all outputs are valid, break
                if len(indices_to_regenerate) == 0:
                    break

            if any([output for output in valid_outputs if output is None]):
                print(
                    f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
                
            return valid_outputs