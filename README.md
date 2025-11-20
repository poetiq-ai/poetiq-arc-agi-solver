# Poetiq: SOTA Reasoning on ARC-AGI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ARC-AGI](https://img.shields.io/badge/Task-ARC--AGI-red)](https://arcprize.org/)

This repository allows reproduction of **Poetiq's** record-breaking submission to the ARC-AGI-1 and ARC-AGI-2 benchmarks.

Full analysis is available in our launch post, **[Traversing the Frontier of Intelligence and Reasoning](https://poetiq.ai/posts/arcagi_announcement/)**.

---

## 📊 Results

<p align="center">
  <img src="arcagi1.png" width="45%" />
  <img src="arcagi2.png" width="45%" />
</p>

## 🛠️ Usage

### Prerequisites
- Python 3.11+
- API Keys for the models you wish to test (Gemini, OpenAI, etc.)

### Quick Start

1. Setup the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a .env file in the root directory. You must include keys for the models you intend to run.

    ```bash
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...
    ```

3. Modify the constants in main.py to set the problem set, number of problems, etc. Then run the script:

    ```bash
    python main.py
    ```

4. By default, the code runs the Poetiq 3 config described in the blog post. You can uncomment other ones or modify the config in config.py

## 📄 Contact
If you use this code or these results in your research, please cite our blog post:

Poetiq Team. (2024). *Traversing the Frontier of Intelligence and Reasoning*. Poetiq AI. [https://poetiq.ai/posts/arcagi_announcement/](https://poetiq.ai/posts/arcagi_announcement/)

For questions or to discuss the future of reasoning, reach out to us at poetiq@poetiq.ai.

[![X (formerly Twitter)](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/poetiq_ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/poetiq/)
