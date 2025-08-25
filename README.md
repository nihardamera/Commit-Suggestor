# Commit Suggester 🤖✍️

Your friendly AI assistant for crafting perfect Git commit messages! This project is a fine-tuned LLM that takes a `git diff` and generates a conventional commit message.

## The Problem

We've all been there: you've just finished a feature, fixed a bug, or made a small tweak. Now it's time to commit your changes, and you're staring at a blank commit message prompt. What do you write? A good commit message is concise, informative, and follows a consistent style. The [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification is a great standard, but it can be tedious to follow manually.

## The Solution

**Commit Suggestor** is a fine-tuned language model that automates this process. Just provide the `git diff` of your changes, and it will generate a commit message for you.

## How it Works

This project uses a pre-trained language model from Hugging Face and fine-tunes it on a dataset of `git diff`s and their corresponding commit messages. The model learns the relationship between the code changes and the commit message, allowing it to generate new messages for new diffs.

### Tech Stack

* **Model:** `mistralai/Mistral-7B-v0.1`
* **Fine-Tuning:** Hugging Face `transformers`, `peft` (for LoRA), and `datasets`.
* **Application:** Gradio for a simple web interface.
* **Data:** A custom dataset of git diffs and commit messages (we'll create this in the `scripts`).

## Getting Started

### Prerequisites

* Python 3.8+
* An NVIDIA GPU for fine-tuning (you can use Google Colab for this)
* A GitHub Personal Access Token (for data mining)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/commit-companion.git](https://github.com/YOUR_USERNAME/commit-companion.git)
    cd commit-companion
    ```

2.  **Create a virtual environment and install the dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Usage

1.  **(Optional) Collect Your Own Data:**
    Before preparing the data, you can run the mining script to collect `diff` and `commit_message` pairs from public GitHub repositories. **Note:** This requires a GitHub Personal Access Token set as an environment variable (`GITHUB_TOKEN`).
    ```bash
    python scripts/00_mine_data.py
    ```

2.  **Prepare the data:**
    Run the data preparation script to create the dataset for fine-tuning.
    ```bash
    python scripts/01_prepare_data.py
    ```

3.  **Fine-tune the model:**
    This will take some time and requires a GPU.
    ```bash
    python scripts/02_fine_tune.py
    ```

4.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
    This will launch a web interface where you can paste your `git diff` and get a commit message.

## Future Improvements

* Integrate with Git hooks to automatically generate a commit message when you run `git commit`.
* Create a VS Code extension for a more integrated experience.
* Experiment with different base models and fine-tuning techniques.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
