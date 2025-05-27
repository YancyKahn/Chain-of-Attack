import uvicorn
import torch
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import argparse
from typing import Dict, List
import gc

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

model_path_config = {
    "princeton-nlp/sup-simcse-roberta-large": "princeton-nlp/sup-simcse-roberta-large",
    "tomh/toxigen_roberta": "tomh/toxigen_roberta",
    "meta-llama/Llama-Guard-3-8B": "meta-llama/Llama-Guard-3-8B"
}

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_path_config[model_name]
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
    def batched_generate_by_thread(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float):
        """
        Generates response by multi-threads for each requests
        """
        raise NotImplementedError

import torch
import gc

class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_path_config[model_name]
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.tokenizer
    

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0):
        try:
            inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

            if temperature > 0:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens, 
                    do_sample=True,
                    temperature=temperature,
                    eos_token_id=self.eos_token_ids,
                    top_p=top_p,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens, 
                    do_sample=False,
                    eos_token_id=self.eos_token_ids,
                    top_p=1,
                    temperature=1, # To prevent warning messages
                )
            
            # If the model is not an encoder-decoder type, slice off the input tokens
            if not self.model.config.is_encoder_decoder:
                output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

            # Batch decoding
            outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        except RuntimeError as e:
            print(e)
            try:
                del inputs
            except:
                pass

            try:
                del output_ids
            except: 
                pass

            gc.collect()
            torch.cuda.empty_cache()

            return [""] * len(full_prompts_list), [None] * len(full_prompts_list)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list


    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            # self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])


def load_tokenizer_and_model(model_name, device):
    model_path = model_path_config[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                use_fast=True,
                                                trust_remote_code=True
                                                )
    
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        device_map=device,
                                                        trust_remote_code=True
                                                        )
    
    if 'llama-2' in model_name.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'vicuna' in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if 'chatglm2' in model_name.lower():
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    if "baichuan" in model_name.lower():
        tokenizer.padding_side = 'left'

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HuggingFace(model_name, model, tokenizer)

    return lm



class SemRelvance():
    def __init__(self, device, model_name = "princeton-nlp/sup-simcse-roberta-large"):
        # roberta-base, princeton-nlp/sup-simcse-roberta-large, sentence-transformers/paraphrase-MiniLM-L12-v2
        model_name = model_path_config[model_name]
        self.model_name = model_name

        self.model, self.tokenizer = self.load_model(device)

    def load_model(self, device):
        print("Loading model: {}".format(self.model_name))
        if "roberta-base" in self.model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaModel.from_pretrained(self.model_name)
        elif "simcse-roberta" in self.model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaModel.from_pretrained(self.model_name)
        elif "hellonlp/simcse-roberta-large-zh" in self.model_name:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
        elif "paraphrase-MiniLM" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        else:
            raise NotImplementedError

        return self.model, self.tokenizer
    
    def encode_text(self, text):
        # Use tokenizer to convert text to input tokens
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        max_tokens = 512
        tokens = tokens[:, :max_tokens]
        # Get the last hidden state from the model
        with torch.no_grad():
            output = self.model(tokens)
        # Extract the last hidden state
        text_representation = output.last_hidden_state.mean(dim=1).squeeze()
        return text_representation


    def compute_similarity(self, text1, text2):
        # Encode the text into embeddings
        representation1 = self.encode_text(text1)
        representation2 = self.encode_text(text2)

        # Compute the cosine similarity between the two vectors
        similarity = torch.nn.functional.cosine_similarity(representation1, representation2, dim=0)
        return similarity.item()


class Toxigen():
    def __init__(self, model_name = "tomh/toxigen_roberta"):
        # roberta-base, princeton-nlp/sup-simcse-roberta-large, sentence-transformers/paraphrase-MiniLM-L12-v2
        model_name = model_path_config[model_name]
        self.model_name = model_name

        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        print("Loading model: {}".format(self.model_name))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self.model, self.tokenizer
    
    def get_toxicity(self, texts):
        try:
            # Use tokenizer to convert text to input tokens
            tokens = self.tokenizer(texts, return_tensors='pt', padding=True)
            # Get the last hidden state from the model
            max_tokens = 512
            tokens = {k: v[:, :max_tokens] for k, v in tokens.items()}
            
            with torch.no_grad():
                output = self.model(**tokens).logits
            # Extract the last hidden state
            prob = torch.nn.functional.softmax(output, dim=1)
            label = torch.argmax(prob, dim=1)
        except Exception as e:
            print(e)
            label = torch.zeros(len(texts))


        result = []

        for i in range(len(texts)):
            result.append({
                "text": texts[i],
                "type": str(label[i].item()),
                "toxicity": prob[i][1].item()
            })

        return result
    
class Judge():
    def __init__(self, device, model_name):
        self.model_name = model_path_config[model_name]
        self.model, self.tokenizer = self.load_model(device)

    def load_model(self, device):
        print("Loading model: {}".format(self.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       use_fast=True, 
                                                       trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                          torch_dtype=torch.float16, 
                                                          device_map=device,
                                                          low_cpu_mem_usage=True,
                                                          trust_remote_code=True)
        return self.model, self.tokenizer

    def llama_guard_judge(self, goal, target_str):
        def moderate(chat):
            input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
            output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
            prompt_len = input_ids.shape[-1]
            return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        return moderate([
            {"role": "user", "content": f"{goal}"},
            {"role": "assistant", "content": f"{target_str}"},
        ]).strip()

def main(args):

    app = FastAPI()
    host = args.host
    port = args.port
    log_level = args.log_level
    lms = {}
    for model_name in args.model_name:
        if "llama-guard" in model_name.lower():
            lms["llama-guard"] = Judge(args.device, model_name)
    
    lms["sem_relevance"] = SemRelvance(args.device)
    # lms["sem_relevance_zh"] = SemRelvance("hellonlp/simcse-roberta-large-zh")
    lms["toxigen"] = Toxigen()

    
    # llama-guard
    if lms.get("llama-guard", None) != None:
        @app.post("/generate/llama-guard")
        async def generate(request: Request):
            # Get request data
            data = await request.json()

            goal = data.get("goal", "")
            target_str = data.get("target_str", "")

            result = lms["llama-guard"].llama_guard_judge(goal, target_str)

            return {"result": result}
    
    if lms.get("sem_relevance", None) != None:
        @app.post("/sem_relevance")
        async def generate(request: Request):
            # Get request data
            data = await request.json()

            text1 = data.get("text1", "")
            text2 = data.get("text2", "")

            similarity = lms["sem_relevance"].compute_similarity(text1, text2)

            return {"similarity": similarity}
    
    if lms.get("sem_relevance_zh", None) != None:
        @app.post("/sem_relevance_zh")
        async def generate(request: Request):
            # Get request data
            data = await request.json()

            text1 = data.get("text1", "")
            text2 = data.get("text2", "")

            similarity = lms["sem_relevance_zh"].compute_similarity(text1, text2)

            return {"similarity": similarity}
    
    if lms.get("toxigen", None) != None:
        @app.post("/toxigen")
        async def generate(request: Request):
            # Get request data
            try:
                data = await request.json()
                texts = data.get("texts", [])


                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:128]
                    texts_rebuild.append(text)
                result = lms["toxigen"].get_toxicity(texts_rebuild)

            except Exception as e:
                # make sure the length of texts less than 512
                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:256]
                    texts_rebuild.append(text)

                result = lms["toxigen"].get_toxicity(texts_rebuild)


            return {"result": result}
    
    
    uvicorn.run(app, host=host, port=port, log_level=log_level)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##### Model Settings #####
    parser.add_argument("--model-name", 
                        type=str,
                        nargs="+",
                        default=["meta-llama/Llama-Guard-3-8B"],
                        help="model_name")

    #### FastAPI Settings ####
    parser.add_argument("--host", 
                        type=str, 
                        default="0.0.0.0", 
                        help="host")
    
    parser.add_argument("--port",
                        type=int,
                        default=9999,
                        help="port")
    
    parser.add_argument("--log_level",
                        type=str,
                        default="info",
                        help="log_level")
    
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="device")
    
    args = parser.parse_args()

    print(args)
    
    main(args)
