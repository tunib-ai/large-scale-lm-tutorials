from transformers import AutoModelForCausalLM, AutoTokenizer
from parallelformers import parallelize

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    parallelize(model, num_gpus=4, fp16=True, verbose="simple")

    inputs = tokenizer(
        "Parallelformers is",
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs,
        num_beams=5,
        no_repeat_ngram_size=4,
        max_length=15,
    )

    print(f"\nOutput: {tokenizer.batch_decode(outputs)[0]}")
