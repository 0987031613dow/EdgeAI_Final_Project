# -*- coding: utf-8 -*-
import torch
from datasets import load_dataset
from gptqmodel import QuantizeConfig, GPTQModel
from transformers import AutoTokenizer


"""GPTQMODEL 4bit quantization"""

def chunked_calibration(texts, tokenizer, block_size=256):
    # 將長文本串接並切成固定長度的 block
    full = "\n\n".join(texts)
    enc = tokenizer(full, return_tensors="pt")
    ids = enc["input_ids"][0]
    chunks = []
    for i in range(0, ids.size(0) - block_size + 1, block_size):
        chunk_ids = ids[i : i + block_size]
        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": torch.ones_like(chunk_ids)
        })
    return chunks


def main():
    # 1. 參數設定
    base_model_id = "ziyingchen1106/Llama-3.2-3B-Instruct-fp16-lora-wikitext"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    output_dir = "./Llama-3.2-3B-Instruct-fp16-lora-gptqmodel-4bit"
    num_calib = 4096      # 校準樣本數
    batch_size = 2        # 視 VRAM 大小調整
    block_size = 256      # 每個 block 長度

    # 2. 載入 Wikitext-2 train split
    print("載入校準資料…")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = ds["text"][:num_calib]

    # 3. 製作 chunked calibration data
    print("製作 chunked calibration data…")
    calibration_data = chunked_calibration(texts, tokenizer, block_size)
    print(f"→ 共 {len(calibration_data)} 個 block，每個 {block_size} tokens")

    # 4. 建立量化設定
    quant_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=True,
        static_groups=False,
        sym=True,
        lm_head=False,
        true_sequential=True,
        quant_method="gptq",
        damp_percent=0.1,
        damp_auto_increment=0.0015,
        device="cuda"
    )

    # 5. 載入原始模型並量化
    print(f"載入原始模型：{base_model_id} …")
    model = GPTQModel.load(base_model_id, quant_config)

    # 確保參數為 float32
    print("將模型參數轉為 float32…")
    for name, param in model.model.named_parameters():
        if param.dtype in (torch.half, torch.bfloat16):
            param.data = param.data.float()
            if param.grad is not None:
                param.grad = param.grad.float()

    print("開始量化（calibration）…")
    model.quantize(
        calibration_data,
        batch_size=batch_size,
        tokenizer=tokenizer
    )

    # 6. 儲存量化後模型
    print(f"儲存量化後模型到：{output_dir} …")
    model.save(output_dir)

    print("量化完成！資料夾可直接用 vLLM/transformers 載入推理。")


if __name__ == "__main__":
    main()