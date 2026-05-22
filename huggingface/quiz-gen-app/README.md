# Quiz Generator

Gradio app that turns a reference URL, uploaded file, or pasted text into Korean learning quizzes.

## Run

```powershell
uv sync
uv run python src/quiz_gen.py
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`, matching the text-generation model already used by the working examples in this repository.

```powershell
uv run huggingface-cli login
```
