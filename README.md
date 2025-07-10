Here is your **enhanced version of the `README.md`**, including **all additional required setup steps** such as **offloading support**, **model pre-download**, **offline mode**, and **folders** like `./offload`.

---

# ✅ Updated `README.md` with All Additional Requirements

```markdown
# Gemma-Vision-Offline

A simple CLI tool to **describe images offline** using the **Gemma 3n multimodal model**. Runs fully offline using a locally downloaded model — no internet required.

---

## 🗭 Overview

This repository provides:

* A script to run Gemma 3n in **Vision Mode**
* Works **offline** on **CPU or GPU**
* Uses HuggingFace `transformers` for inference
* Supports **model offloading to disk** via `accelerate`

---

## 💻 Minimum Hardware & Software Requirements

| Component      | Minimum Requirement                    |
| -------------- | -------------------------------------- |
| CPU            | Quad-core ARM or x86                   |
| RAM            | 32 GB (recommended)                    |
| Disk Space     | 10–15 GB (for model, cache, offload)   |
| GPU (optional) | NVIDIA with 8 GB VRAM (if using CUDA)  |
| OS             | Ubuntu 20.04+ (Tested on 22.04)        |
| Python Version | Python 3.9 or later                    |
| Virtual Env    | Recommended (`venv` or `conda`)        |

---

## 📦 Repository Structure

```

Gemma-Vision-Offline/
├── gemma\_local\_model/       ← Pre-downloaded model files
├── describe\_image.py        ← Main CLI script
├── requirements.txt         ← Required Python libraries
├── offload/                 ← Disk offload folder (created manually)
└── README.md                ← This file

````

---

## ⚙️ Setup & Installation

### 1. **Clone the repository**

```bash
git clone https://github.com/yourusername/gemma-vision-offline.git
cd gemma-vision-offline
````

### 2. **Prepare a Python virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### 4. **Download and save the model locally (one-time)**

This step ensures offline mode will work.

```python
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("google/gemma-3b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3b-it")

processor.save_pretrained("./gemma_local_model")
model.save_pretrained("./gemma_local_model")
```

Or from terminal:

```bash
python download_model.py
```

*(You can create a `download_model.py` script if needed)*

---

### 5. **Create an offload folder**

This is required for memory-efficient execution with `accelerate`.

```bash
mkdir -p offload
```

---

## 🚀 How to Use

```bash
python describe_image.py
```

You'll see:

```
[Gemma Vision (Offline)]
Enter the path to your image (.jpg/.png): bee.jpg
Enter a prompt to guide the description: What's happening in this photo?
🤖 The model says: "A bee is landing on a flower..."
```

---

## 📜 Script: `describe_image.py`

```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Load local model
processor = AutoProcessor.from_pretrained("./gemma_local_model")
model = AutoModelForCausalLM.from_pretrained(
    "./gemma_local_model",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    offload_folder="./offload",              # enable offload
    offload_state_dict=True                  # move model weights to disk if needed
)

def main():
    print("[Gemma Vision (Offline)]")
    img_path = input("Enter the path to your image (.jpg/.png): ").strip()
    prompt = input("Enter a prompt to guide the description: ").strip()
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Failed to open image:", e)
        return

    formatted_prompt = f"<image_soft_token> {prompt}"
    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    print("🤖", response)

if __name__ == "__main__":
    main()
```

---

## ✅ Requirements (`requirements.txt`)

```
transformers==4.53.0      # core model + processor support
torch                     # tensor and model inference
accelerate                # device placement / offloading
Pillow                    # image loading and handling
numpy                     # array and tensor operations
timm                      # image model components (vision tower support)
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📴 Offline Mode

```bash
HF_HUB_OFFLINE=1 python describe_image.py
```

> Ensure `gemma_local_model/` and `offload/` are present and populated.

---

## 🙏 Credits

* **Gemma 3n Model** by [Google DeepMind](https://deepmind.google)
* Built with [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

## 📌 License

This project is released under the MIT License.

```
