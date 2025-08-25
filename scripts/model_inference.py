import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from data_preparing import create_prompt

def run_inference(diff):
    model_path = "./commit-suggestor-model"

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

    prompt = create_prompt(diff)
    result = pipe(prompt)
    print(result[0]['generated_text'])

if __name__ == "__main__":
    example_diff = """
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,5 +1,5 @@
 import gradio as gr
 
 def greet(name):
-    return "Hello " + name + "!"
+    return "Greetings, " + name + "!"
 
 iface = gr.Interface(fn=greet, inputs="text", outputs="text")
 iface.launch()
    """
    run_inference(example_diff)
