<h2 style="color:red;">✅ Ollama</h2>


<h3 style="color:blue;">📌 What is Ollama?</h3>

**Ollama** is a tool and platform that makes it easy to **run and interact with large language models (LLMs) locally on your computer,** especially models like ```LLaMA```, ```Mistral```, ```Gemma```, and others — **without needing cloud infrastructure or APIs like OpenAI's**.

<h3 style="color:blue;">🔧 What Does Ollama Do?</h3>

- **Runs LLMs locally:** No internet connection or cloud server needed after downloading a model.

- **Supports popular open-source models:** Includes LLaMA 2/3, Mistral, Gemma, Code LLaMA, and others.

- **Simple CLI tool:** You can start chatting with a model using the command line.

- **Integrates with apps:** Ollama can serve models through an HTTP API, enabling developers to build local AI applications (e.g., with LangChain, CrewAI, etc.).

<h3 style="color:blue;">✅ Key Features</h3>

| Feature                   | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| **Local model execution** | Uses your own CPU/GPU to run models                                |
| **Model management**      | Download, update, and remove models easily                         |
| **Privacy-first**         | No data is sent to external servers unless you choose to           |
| **Cross-platform**        | Works on macOS, Windows, and Linux                                 |
| **Built-in server**       | Starts a local server (`localhost:11434`) to access models via API |


<h3 style="color:blue;">🧠 Example Usage (Terminal)</h3>

```
ollama run llama3
```

<h3 style="color:blue;">🧠 Example Usage with a prompt</h3>

```
ollama run mistral:7b-instruct
> What's the capital of France?
```

<h3 style="color:blue;">✅ 1. LLaMA Family</h3>

| Model        | Description                                           |
| ------------ | ----------------------------------------------------- |
| `llama2`     | Meta's LLaMA 2, general-purpose LLM (7B, 13B, 70B)    |
| `llama3`     | Meta's latest, better quality and reasoning (8B, 70B) |
| `llama3:8b`  | Smaller, fast and efficient                           |
| `llama3:70b` | Very powerful, large model                            |


<h3 style="color:blue;">✅ 2. Mistral Models</h3>

| Model          | Description                                            |
| -------------- | ------------------------------------------------------ |
| `mistral`      | Open-weight, performant 7B model                       |
| `mixtral`      | Mixture-of-experts (MoE) version (12.9B active params) |
| `mixtral:8x7b` | Specific variant of Mixtral                            |


<h3 style="color:blue;">✅ 3. Gemma</h3>

| Model      | Description                         |
| ---------- | ----------------------------------- |
| `gemma`    | Google's open model, Gemma (2B, 7B) |
| `gemma:2b` | Lightweight and fast                |
| `gemma:7b` | More powerful                       |

<h3 style="color:blue;">✅ 4. Phi</h3>

| Model   | Description                           |
| ------- | ------------------------------------- |
| `phi`   | Microsoft's small, high-quality model |
| `phi:2` | Updated Phi-2 model                   |

<h3 style="color:blue;">✅ 5. Code LLMs</h3>

| Model           | Description                         |
| --------------- | ----------------------------------- |
| `codellama`     | Meta's LLaMA variant for code tasks |
| `codellama:7b`  | LLaMA 2 based code model (7B)       |
| `codellama:13b` | Larger version (13B)                |

<h3 style="color:blue;">✅ 6. Neural Chat</h3>

| Model         | Description                         |
| ------------- | ----------------------------------- |
| `neural-chat` | Fine-tuned for conversational tasks |

<h3 style="color:blue;">✅ 7. OpenChat</h3>

| Model      | Description                                       |
| ---------- | ------------------------------------------------- |
| `openchat` | Fine-tuned model with great instruction following |

<h3 style="color:blue;">✅ 8. Dolphin</h3>

| Model             | Description                                  |
| ----------------- | -------------------------------------------- |
| `dolphin-mixtral` | Fine-tuned Mixtral model, very chat-friendly |

<h3 style="color:blue;">✅ 9. LLaVA (Multimodal)</h3>

| Model   | Description                                 |
| ------- | ------------------------------------------- |
| `llava` | Vision + language model, image + text input |

<h3 style="color:blue;">✅ 10. Solar</h3>

| Model   | Description                         |
| ------- | ----------------------------------- |
| `solar` | Compact model with high performance |

<h3 style="color:blue;">✅ 11. Starling</h3>

| Model      | Description                           |
| ---------- | ------------------------------------- |
| `starling` | Reward model fine-tuned for alignment |


<h3 style="color:blue;">✅ 12. TinyLLaMA</h3>

| Model       | Description                      |
| ----------- | -------------------------------- |
| `tinyllama` | Super lightweight version (1.1B) |


<h3 style="color:blue;">✅ 13. Yi</h3>

| Model | Description                          |
| ----- | ------------------------------------ |
| `yi`  | Open Chinese-English bilingual model |



<h3 style="color:blue;">🔍 To List All Models from CLI:</h3>

```
ollama list
```

<h3 style="color:blue;">📥 To Pull a Model:</h3>

```
ollama pull llama3
```

<h3 style="color:blue;">🧠 To Run a Model:</h3>

```
ollama run llama3
```

<h3 style="color:blue;">🖥️ Hardware Requirements by Model Size</h3>

| **Model Size**       | **Recommended RAM** | **Recommended VRAM (GPU)**   | **CPU**                         | **Disk Space** | **Notes**                                      |
| -------------------- | ------------------- | ---------------------------- | ------------------------------- | -------------- | ---------------------------------------------- |
| Tiny (1B–3B)         | ≥ 8 GB              | Optional (2–4 GB VRAM)       | 4-core x86 CPU                  | \~10 GB        | Runs on CPU, slow on older machines            |
| Small (7B)           | ≥ 16 GB             | ≥ 4–6 GB VRAM                | 6-core, modern CPU              | \~15–20 GB     | Most popular size, runs well on modern systems |
| Medium (13B)         | ≥ 24 GB             | ≥ 8 GB VRAM                  | 8-core or better                | \~25–30 GB     | Slower on CPU; GPU recommended                 |
| Large (30B)          | ≥ 32 GB             | ≥ 16 GB VRAM                 | 8-core+, AVX512 support helpful | ≥ 50 GB        | GPU required for real-time chat                |
| Very Large (65B–70B) | ≥ 64 GB             | ≥ 32 GB VRAM (e.g. RTX 4090) | High-end workstation/server CPU | ≥ 70–100 GB    | For advanced users; long startup time on CPU   |


<h3 style="color:blue;">🧰 Software Requirements</h3>

| **Component**               | **Requirement**                                                                |                      |
| --------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| **Operating System**        | macOS (Intel/Apple Silicon), Linux, Windows (via WSL2 or native GUI on Win 11) |                      |
| **Installation**            | Via CLI (\`curl [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh\`) or Windows GUI |
| **CUDA (GPU)**              | NVIDIA GPU support for acceleration (CUDA >= 11.7)                             |                      |
| **Optional GPU Config**     | `OLLAMA_FLASH_ATTENTION=1` to enable flash attention for faster decoding       |                      |
| **Supported Architectures** | x86-64, Apple M1/M2/M3 (ARM64 supported for macOS)                             |                      |
| **Environment**             | Works with Docker, CLI, or as a backend for LangChain, Python, etc.            |                      |


<h3 style="color:blue;">📦 Model & Storage Considerations</h3>

| **Model Name**        | **Typical Size (Quantized)** | **Model Types**            | **Use Case**                       |
| --------------------- | ---------------------------- | -------------------------- | ---------------------------------- |
| `tinyllama`           | \~1–2 GB                     | Chat/General               | Very fast, low resource            |
| `phi`, `neural-chat`  | \~2–3 GB                     | Chat, Conversational       | Efficient for local use            |
| `llama2:7b`           | \~4–5 GB                     | General-purpose            | Best balance of size & performance |
| `mistral`, `gemma:7b` | \~4–6 GB                     | Chat, RAG, QA              | Faster and newer alternatives      |
| `codellama:13b`       | \~7–8 GB                     | Code generation            | GPU highly recommended             |
| `llama3:70b`          | \~40–45 GB                   | Top-tier reasoning and RAG | Only suitable for high-end systems |


<h3 style="color:blue;">⚙️ Performance Benchmarks (Approx.)</h3>

| **Model**    | **Cold Load Time (CPU)** | **Cold Load Time (GPU)** | **Token Generation Speed**             |
| ------------ | ------------------------ | ------------------------ | -------------------------------------- |
| `llama2:7b`  | 15–25 sec                | 3–5 sec                  | \~5–15 tokens/sec (CPU), \~40–70 (GPU) |
| `mistral`    | 10–20 sec                | 2–4 sec                  | Fast, efficient                        |
| `llama3:70b` | 60–90 sec                | 5–10 sec (RTX 4090)      | Very fast, but huge size               |


<h3 style="color:blue;">🧠 CodeLlama Models Overview</h3>

| **Model Name**  | **Parameters** | **Quantized Size** | **Use Case**                        | **Model Type**                 |
| --------------- | -------------- | ------------------ | ----------------------------------- | ------------------------------ |
| `codellama`     | 7B             | \~4–6 GB           | General code generation             | LLaMA 2 base + code fine-tuned |
| `codellama:7b`  | 7B             | \~4–6 GB           | Efficient for local code completion | Standard                       |
| `codellama:13b` | 13B            | \~7–9 GB           | More context, better accuracy       | Larger version                 |

<h3 style="color:blue;">🖥️ Hardware Requirements</h3>

| **Model**       | **RAM (Recommended)** | **GPU VRAM (Recommended)** | **CPU (Min)**       | **Disk Space** |
| --------------- | --------------------- | -------------------------- | ------------------- | -------------- |
| `codellama`     | ≥ 16 GB               | Optional, ≥ 4 GB           | 4–6 core x86        | \~6 GB         |
| `codellama:7b`  | ≥ 16 GB               | ≥ 6 GB                     | 6-core modern CPU   | \~8 GB         |
| `codellama:13b` | ≥ 24 GB               | ≥ 8–12 GB                  | 8-core+ recommended | \~12 GB        |


**📝 Note:** GPU is highly recommended for codellama:13b, especially for low-latency completions.

<h3 style="color:blue;">⚙️ Software Requirements</h3>

| **Component**     | **Details**                                    |
| ----------------- | ---------------------------------------------- |
| OS                | macOS, Linux, Windows (WSL2 or GUI for Win11)  |
| GPU Support       | NVIDIA CUDA 11.7+ (if using GPU acceleration)  |
| Ollama CLI/GUI    | `ollama pull codellama:7b` or `:13b`           |
| Usage             | CLI or API (`ollama run codellama:7b`)         |
| Integration Ready | Works with LangChain, VS Code extensions, etc. |


<h3 style="color:blue;">🧪 Performance Comparison (Approx.)</h3>

| **Metric**            | `codellama:7b`           | `codellama:13b`           |
| --------------------- | ------------------------ | ------------------------- |
| Load Time (CPU)       | \~10–20 sec              | \~30–45 sec               |
| Load Time (GPU)       | \~2–5 sec (6GB+)         | \~6–10 sec (12GB+)        |
| Tokens/sec (CPU)      | \~8–15                   | \~4–8                     |
| Tokens/sec (GPU)      | \~50–80                  | \~40–60                   |
| Max Context (default) | 4K tokens (configurable) | 4K–8K depending on tuning |


<h3 style="color:blue;">🧑‍💻 Ideal Use Cases</h3>

| **Model**       | **Recommended For**                                                           |
| --------------- | ----------------------------------------------------------------------------- |
| `codellama`     | Lightweight coding tasks, embedded in apps                                    |
| `codellama:7b`  | Local code assistants, pair programming, code explanations                    |
| `codellama:13b` | Advanced coding help, long function generation, full project structure output |

<h3 style="color:blue;">🧠 Understanding Model Sizes</h3>

The terms **7B** and **13B** refer to the number of parameters in the model — a critical aspect that affects performance, resource requirements, and capability.

| **Model Name** | **Parameter Count** | **Meaning**                                                |
| -------------- | ------------------- | ---------------------------------------------------------- |
| `1B`           | 1 Billion           | Very lightweight, fast, but limited capability             |
| `3B`           | 3 Billion           | Small model for basic tasks                                |
| `7B`           | 7 Billion           | Good balance of speed and performance                      |
| `13B`          | 13 Billion          | Higher accuracy and reasoning; needs more resources        |
| `30B`          | 30 Billion          | Very high capability; slow on CPU                          |
| `70B`          | 70 Billion          | State-of-the-art performance; needs high-end GPU or server |


<h3 style="color:blue;">⚖️ 7B vs 13B – Tradeoffs</h3>

| Feature                    | **7B Model**                     | **13B Model**                        |
| -------------------------- | -------------------------------- | ------------------------------------ |
| **Model Size (quantized)** | \~4–6 GB                         | \~7–9 GB                             |
| **Performance**            | Fast and responsive              | Slower but smarter                   |
| **Memory Usage (RAM)**     | 16 GB minimum                    | 24–32 GB recommended                 |
| **GPU (VRAM)**             | ≥ 6 GB recommended               | ≥ 10–12 GB preferred                 |
| **Accuracy**               | Good for most tasks              | Better understanding and generation  |
| **Context Window**         | Up to 4K tokens                  | Up to 8K tokens (configurable)       |
| **Best For**               | Lightweight apps, fast chat/code | Complex coding, long-form generation |
| **Startup Time**           | 2–5 sec (GPU), 10–20 sec (CPU)   | 5–10 sec (GPU), 30–45 sec (CPU)      |


<h3 style="color:blue;">🧑‍💻 Example Ollama Models by Size</h3>

| **Model**            | **Size** | **Use Case**                             |
| -------------------- | -------- | ---------------------------------------- |
| `tinyllama`          | 1.1B     | Very fast, low-memory devices            |
| `phi`, `neural-chat` | 2–3B     | General chat, embedded systems           |
| `llama2:7b`          | 7B       | Fast, general-purpose LLM                |
| `codellama:7b`       | 7B       | Efficient code assistant                 |
| `codellama:13b`      | 13B      | Advanced code generation & understanding |
| `llama3:70b`         | 70B      | SOTA performance, massive reasoning      |

<h3 style="color:blue;">✅ When to Choose What</h3>

| **If you want...**                 | **Choose**           |
| ---------------------------------- | -------------------- |
| Fast response, low resource usage  | `7B`                 |
| More accurate coding and reasoning | `13B`                |
| Max accuracy and deep knowledge    | `30B` or `70B`       |
| Ultra-lightweight chatbot          | `phi` or `tinyllama` |


<h3 style="color:blue;">⚖️ CodeLlama: 7B vs 13B for Code Tasks</h3>

| Feature                          | `codellama:7b`                            | `codellama:13b`                              |
| -------------------------------- | ----------------------------------------- | -------------------------------------------- |
| **Parameter Count**              | 7 Billion                                 | 13 Billion                                   |
| **Model Size (Quantized)**       | \~4–6 GB                                  | \~7–9 GB                                     |
| **RAM Required (Minimum)**       | 16 GB (bare minimum)                      | 24–32 GB recommended                         |
| **GPU VRAM (Recommended)**       | ≥ 6 GB (good performance)                 | ≥ 10–12 GB (critical for usable performance) |
| **CPU-only Load Time**           | 10–20 seconds                             | 30–45 seconds                                |
| **GPU Load Time**                | 2–5 seconds                               | 6–10 seconds                                 |
| **Token Generation Speed (CPU)** | \~10–20 tokens/sec                        | \~5–8 tokens/sec                             |
| **Token Generation Speed (GPU)** | \~40–80 tokens/sec                        | \~30–60 tokens/sec                           |
| **Context Window**               | \~4K tokens                               | \~4K–8K tokens                               |
| **Code Generation Quality**      | ✅ Fast, decent completions                | ✅✅ More accurate, better long completions    |
| **Handles Complex Prompts**      | Basic–Intermediate                        | Intermediate–Advanced                        |
| **Multilingual Code Support**    | Yes                                       | Yes                                          |
| **Fine-tuned For**               | Python, JavaScript, C++, TypeScript, etc. | Same, with better long-form understanding    |
| **Use Cases**                    | Code snippets, function completions       | Long functions, class design, full solutions |
| **Real-world Example**           | Autocompletes a Python function           | Writes full Flask backend with endpoints     |


<h3 style="color:blue;">🧠 Which One Should You Choose?</h3>

| **Your Setup**                                       | **Recommended Model**        |
| ---------------------------------------------------- | ---------------------------- |
| CPU-only, 16 GB RAM or low VRAM GPU (≤ 6 GB)         | ✅ `codellama:7b`             |
| Decent GPU (e.g., RTX 3060+, ≥10 GB VRAM), 32 GB RAM | ✅✅ `codellama:13b`           |
| High-end GPU (e.g., RTX 4090), 64+ GB RAM            | 🚀 `codellama:13b` or larger |
| Want faster generation, small project completions    | ✅ `7b`                       |
| Need higher accuracy, deeper code reasoning          | ✅✅ `13b`                     |



<h3 style="color:blue;">🧠 Let's compare codellama:7b vs codellama:13b on a real-world coding task.</h3>

<h3 style="color:blue;">👨‍💻 Prompt</h3>

```
"Write a Python Flask REST API with two endpoints: /health returning {'status': 'ok'} and /predict that accepts JSON input { 'age': 30, 'income': 50000 } and returns a dummy prediction { 'approved': true }."
```

<h3 style="color:blue;">⚙️ CodeLlama:7B — Sample Output</h3>

(Faster response, but less structured and minimal error handling)

```
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data.get('age')
    income = data.get('income')
    # dummy logic
    if age > 25 and income > 40000:
        result = {'approved': True}
    else:
        result = {'approved': False}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

**✅ Strengths:** Quick, to the point

**⚠️ Limitations:** No input validation, no error handling, no docstrings.

<h3 style="color:blue;">🔍 CodeLlama:13B — Sample Output</h3>

(Slower response, but more robust and detailed)

```
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate input
        if 'age' not in data or 'income' not in data:
            return jsonify({'error': 'Missing age or income'}), 400

        age = data['age']
        income = data['income']

        # Dummy prediction logic
        approved = age >= 25 and income >= 40000
        return jsonify({'approved': approved}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**✅✅ Strengths:** Adds validation, error handling, and cleaner structure

**💡 Extras:** Can also add Swagger, Pydantic, or type hints if prompted

<h3 style="color:blue;">⚔️ Comparison Summary</h3>

| Feature            | `codellama:7b`    | `codellama:13b`                             |
| ------------------ | ----------------- | ------------------------------------------- |
| **Response Time**  | ⚡ Fast            | 🕒 Slower                                   |
| **Code Quality**   | 🟨 Basic logic    | ✅ Robust & production-grade                 |
| **Validation**     | ❌ None            | ✅ Basic validation added                    |
| **Error Handling** | ❌ None            | ✅ Includes `try-except` block               |
| **Best For**       | Quick prototyping | Realistic codebases or full-stack scaffolds |



<h3 style="color:blue;">1. Another Prompt Comparison (Docker)</h3>

**Prompt:**

```
Write a Dockerfile for a Python Flask app with two endpoints: /health returning {'status': 'ok'} and /predict accepting JSON and returning dummy prediction {'approved': true}."
```

<h3 style="color:blue;">CodeLlama:7B — Sample Output</h3>

```
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Flask app** (```app.py```) code may be very basic, probably missing error handling or requirements details.

**Strengths:** Quick and straightforward

**Limitations:** Lacks automation or context-specific optimizations


<h3 style="color:blue;">CodeLlama:13B — Sample Output</h3>

```
# Use an official lightweight Python runtime
FROM python:3.10-alpine

# Set work directory
WORKDIR /usr/src/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose necessary port
EXPOSE 5000

# Start Flask server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**Flask app** (```app.py```) likely includes validation, error handling, and production readiness.

**Strengths:** More production-ready, efficient, and secure.

<h3 style="color:blue;">2. Automatically Compare Outputs with Python</h3>

You can compare generated code from two models using Python, using tools like ```difflib```. Here's a small example:

```
import difflib

code_7b = """..."""  # insert 7B generated code
code_13b = """..."""  # insert 13B generated code

for line in difflib.unified_diff(code_7b.splitlines(keepends=True),
                                  code_13b.splitlines(keepends=True),
                                  fromfile='7B', tofile='13B'):
    print(line)
```

