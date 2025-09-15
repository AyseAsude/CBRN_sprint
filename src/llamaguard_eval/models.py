import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llama_guard(checkpoint_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """
    Load Llama Guard 3 model from checkpoint path.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map=device
    )
    return tokenizer, model

def classify_instruction_answer(tokenizer, model, instruction: str, answer: str, max_new_tokens: int = 100):
    """
    Use Llama Guard 3 to classify (instruction + answer).
    """
    chat = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer}
    ]

    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, pad_token_id=0)

    prompt_len = input_ids.shape[-1]
    decoded = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    return decoded
 
