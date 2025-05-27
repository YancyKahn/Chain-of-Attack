import os

LANGUAGE = "en"

OPEN_SOURCE_MODEL_API_TOXIGEN = "http://0.0.0.0:9999/toxigen"
OPEN_SOURCE_MODEL_API_SIMCSE = "http://0.0.0.0:9999/sem_relevance"
OPEN_SOURCE_MODEL_API_SIMCSE_CN = "http://0.0.0.0:9999/sem_relevance_zh"
OPEN_SOURCE_MODEL_API_LLAMA_GUARD = "http://0.0.0.0:9999/generate/llama-guard"

ONE_API_BASE = "Your-API-Base-URL"
ONE_API_KEY = "Your-API-Key"

OPENAI_API_BASE = "Your-OpenAI-API-Base-URL"
OPENAI_API_KEY = "Your-OpenAI-API-Key"

ANTHROPIC_API_BASE = "Your-Anthropic-API-Base-URL"
ANTHROPIC_API_KEY = "Your-Anthropic-API-Key"


# CoA experiment parameters setting
ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1
JUDGE_TEMP = 1
JUDGE_TOP_P = 0.9

# MODEL_NAME and MODEL_TEMPLATE
MODEL_CONFIGS= {
            "vicuna-api": ("vicuna_v1.1", "vicuna_v1.1"),
            "llama2-api": ("llama-2", "llama-2"),
            "chatglm-api": ("chatglm", "chatglm"),
            "chatglm2-api": ("chatglm-2", "chatglm-2"),
            "phi2-api": ("phi2", "phi2"),
            "zephyr-api": ("zephyr", "zephyr"),
            "baichuan-api": ("baichuan2-chat", "baichuan2-chat"),
            "one-shot": ("one_shot", "one_shot"),
            "zhipu": ("zhipu", "zhipu"),
            "douyin": ("douyin", "douyin"),
            "wenxinyiyan": ("wenxinyiyan", "wenxinyiyan"),
            "kuaishou": ("kuaishou", "kuaishou"),
            "baichuan": ("baichuan", "baichuan"),
            "zero-shot": ("zero_shot", "zero_shot"),
            "airoboros-1": ("airoboros_v1", "airoboros_v1"),
            "airoboros-2": ("airoboros_v2", "airoboros_v2"),
            "airoboros-3": ("airoboros_v3", "airoboros_v3"),
            "koala-1": ("koala_v1", "koala_v1"),
            "alpaca": ("alpaca", "alpaca"),
            "chatglm": ("chatglm", "chatglm"),
            "chatglm-2": ("chatglm-2", "chatglm-2"),
            "dolly-v2": ("dolly_v2", "dolly_v2"),
            "oasst-pythia": ("oasst_pythia", "oasst_pythia"),
            "oasst-llama": ("oasst_llama", "oasst_llama"),
            "tulu": ("tulu", "tulu"),
            "stablelm": ("stablelm", "stablelm"),
            "baize": ("baize", "baize"),
            "chatgpt": ("chatgpt", "chatgpt"),
            "bard": ("bard", "bard"),
            "falcon": ("falcon", "falcon"),
            "baichuan-chat": ("baichuan_chat", "baichuan_chat"),
            "baichuan2-chat": ("baichuan2_chat", "baichuan2_chat"),
            "falcon-chat": ("falcon_chat", "falcon_chat"),
            "gpt-4": ("gpt-4", "gpt-4"),
            "gpt-4o-mini": ("gpt-4o-mini", "gpt-4o-mini"),
            "gpt-4o-mini-2024-07-18": ("gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"),
            "gpt-4o": ("gpt-4o", "gpt-4o"),
            "gpt-4-turbo": ("gpt-4-turbo", "gpt-4-turbo"),
            "gpt-3.5-turbo": ("gpt-3.5-turbo", "gpt-3.5-turbo"),
            "text-davinci-003": ("text-davinci-003", "text-davinci-003"),
            "gpt-3.5-turbo-instruct": ("gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct"),
            "vicuna": ("vicuna-api", "vicuna_v1.1"),
            "llama2-chinese": ("llama2_chinese", "llama2_chinese"),
            "llama-2": ("llama2-api", "llama-2"),
            "claude-instant-1": ("claude-instant-1", "claude-instant-1"),
            "claude-2": ("claude-2", "claude-2"),
            "palm-2": ("palm-2", "palm-2"),
            "oneapi-llamaguard-3": ("groq-llama-guard-3-8b", "gpt-4o"),
            "oneapi-llama-2-7b": ("qianfan-Llama-2-7B-Chat", "gpt-4o"),
            "oneapi-llama-3-8b": ("dashscope-llama3-8b-instruct", "gpt-4o"),
            "oneapi-mixtral-8x7b": ("cloudflare-mistral-7b-instruct-v0.1", "gpt-4o"),
            "oneapi-gemma-7b": ("cloudflare-gemma-7b-it-lora", "gpt-4o"),
            "oneapi-llama-3.1-8b": ("siliconflow-meta-llama-meta-llama-3.1-8b-instruct", "gpt-4o"),
            "oneapi-llama3.1-405b": ("dashscope-llama3.1-405b-instruct", "gpt-4o"),
            "oneapi-llama3-70b": ("dashscope-llama3-70b-instruct", "gpt-4o"),
            "oneapi-llama3.1-70b": ("dashscope-llama3.1-70b-instruct", "gpt-4o"),
            "oneapi-qwen2.5-7b": ("dashscope-qwen2.5-7b-instruct", "gpt-4o"),
            "oneapi-qwen-turbo": ("dashscope-qwen-turbo", "gpt-4o"),
            "oneapi-qwen-plus": ("dashscope-qwen-plus", "gpt-4o"),
            "oneapi-qwen-max": ("dashscope-qwen-max", "gpt-4o"),
            "oneapi-qwen-plus-0919": ("dashscope-qwen-plus-0919", "gpt-4o"),
            "oneapi-qwen-plus-1125": ("dashscope-qwen-plus-1125", "gpt-4o"),
            "oneapi-yi-large-turbo": ("dashscope-yi-large-turbo", "gpt-4o"),
            "oneapi-biachuan2-7b": ("dashscope-baichuan2-7b-chat-v1", "gpt-4o"),
            "oneapi-gpt-4o-mini": ("custom-gpt-4o", "gpt-4o-mini"),
            "oneapi-gpt-4o": ("custom-gpt-4o", "gpt-4o"),
            "oneapi-claude-3-5-sonnet": ("my-claude-3-5-sonnet-latest", "gpt-4o"),
            "oneapi-claude-3-haiku-20240307": ("my-claude-3-haiku-20240307", "gpt-4o"),
            "oneapi-gemini-1.5-flash": ("gemini-gemini-1.5-flash", "gpt-4o")
        }