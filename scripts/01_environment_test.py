"""
scripts/01_environment_test.py
Sanity-check the environment before running experiments.
Run this first and share the console output before proceeding.

Usage:
    python scripts/01_environment_test.py
"""

import sys
import platform


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def check_python() -> None:
    v = sys.version_info
    if v.major == 3 and v.minor >= 9:
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python 3.9+ required, found {v.major}.{v.minor}")


def check_imports() -> None:
    required = [
        ("torch",               "PyTorch"),
        ("transformers",        "Hugging Face Transformers"),
        ("datasets",            "Hugging Face Datasets"),
        ("sentence_transformers","Sentence Transformers"),
        ("numpy",               "NumPy"),
        ("pandas",              "Pandas"),
        ("scipy",               "SciPy"),
        ("sklearn",             "Scikit-Learn"),
        ("matplotlib",          "Matplotlib"),
        ("seaborn",             "Seaborn"),
        ("tqdm",                "tqdm"),
    ]
    for module, label in required:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "?")
            _ok(f"{label} {version}")
        except ImportError:
            _fail(f"{label} not installed — run: pip install -r requirements.txt")


def check_torch_device() -> None:
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"CUDA device: {name} ({mem:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _ok("Apple MPS (Metal) device available")
    else:
        _ok("CPU only — experiments will be slower (~3h). GPU strongly recommended.")


def check_triviaqa_sample() -> None:
    from datasets import load_dataset
    print("  Loading 5-example TriviaQA sample …")
    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation[:5]")
        assert len(ds) == 5
        q0 = ds[0]["question"]
        _ok(f"TriviaQA loaded. Sample Q: '{q0[:60]}…'")
    except Exception as e:
        _fail(f"TriviaQA load failed: {e}")


def check_gpt2_inference() -> None:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    print("  Loading gpt2-medium (may download ~1.5 GB on first run) …")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.eval()
    prompt = "Q: What is the capital of France?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    _ok(f"gpt2-medium inference: '{answer.strip()}'")


def check_distilgpt2_inference() -> None:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    prompt = "Q: What is the capital of Germany?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    _ok(f"distilgpt2 inference: '{answer.strip()}'")


def check_sentence_embeddings() -> None:
    from sentence_transformers import SentenceTransformer
    print("  Loading all-MiniLM-L6-v2 (may download ~90 MB on first run) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["What is the capital of France?"])
    assert emb.shape == (1, 384), f"Unexpected shape: {emb.shape}"
    _ok(f"Sentence embedding shape: {emb.shape} (expected (1, 384))")


def main() -> None:
    print("=" * 55)
    print("  Environment Sanity Check — AITR Lab 02 (MRA14)")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 55)

    check_python()
    check_imports()
    check_torch_device()
    check_triviaqa_sample()
    check_gpt2_inference()
    check_distilgpt2_inference()
    check_sentence_embeddings()

    print("\n" + "=" * 55)
    print("  All checks passed. Environment is ready.")
    print("  Next: python scripts/02_run_baseline.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
