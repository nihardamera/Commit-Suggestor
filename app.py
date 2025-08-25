import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_path = "./commit-suggestor-model"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

def create_prompt(diff):
    return f"""Given the following git diff, generate a conventional commit message.
### Git Diff:
{diff}
### Commit Message:
"""

def generate_commit_message(diff):
    if not diff:
        return "Please provide a git diff."

    prompt = create_prompt(diff)
    result = pipe(prompt)

    generated_text = result[0]['generated_text']
    commit_message = generated_text.split("### Commit Message:")[1].strip()
    return commit_message


iface = gr.Interface(
    fn=generate_commit_message,
    inputs=gr.Textbox(lines=20, placeholder="Paste your git diff here..."),
    outputs="text",
    title="Commit Suggestor 🤖✍️",
    description="Your friendly AI assistant for crafting perfect Git commit messages! Paste your `git diff` below and get a conventional commit message.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
