# Project Setup and Run Guide

This guide explains how to set up and run the project using **uv** for environment management and **Ollama** for local LLMs.

## 1. Install `uv`

### Windows (PowerShell)
```powershell
iwr https://astral.sh/uv/install.ps1 -useb | iex
```
Close and reopen the terminal after installation.

### Linux / macOS (bash/zsh)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your shell so that `uv` is on the `PATH`.

Verify:
```bash
uv --version
```

## 2. Install Ollama

### Windows
Download and install the Ollama installer from the official site, then launch **Ollama**.

### macOS (Homebrew)
```bash
brew install ollama
ollama serve
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
# Start the service (method may vary by distro)
ollama serve
```

Verify:
```bash
ollama --version
```

> **Note:** Ensure the Ollama service is running before starting the app.

## 3. Clone the repository
```bash
git clone https://github.com/wooihaw/sea_turtle_rag_chatbot.git
cd sea_turtle_rag_chatbot
```

## 4. Restore the environment and dependencies
```bash
uv sync
```

This command creates/uses an isolated environment and installs all dependencies declared by the project.

## 5. Run the application
```bash
uv run streamlit run main.py
```

The app will open in your browser. Default URL is printed by Streamlit (commonly `http://localhost:8501`).

## 6. First run model downloads

On first run, the application or Ollama may download one or more models. This can take time depending on your network speed and disk space. Do not interrupt the process.

### Optional: Pre-pull a model
If your app expects a specific model, you can pre-download it:
```bash
ollama pull llama3
```
Replace `llama3` with the required model name if different.

---

## Troubleshooting

- **`uv: command not found`**: Restart the terminal, or confirm that your shell `PATH` includes the directory printed by the installer.
- **Ollama not reachable**: Ensure `ollama serve` is running and not blocked by a firewall.
- **Port already in use**: Start Streamlit on another port:
  ```bash
  uv run streamlit run main.py --server.port 8502
  ```
- **Model not found**: Pull the model explicitly with `ollama pull <model>` and rerun the app.

## Uninstall (optional)

- **uv**: Remove the installed binaries from your user installation directory. See the `uv` documentation for your OS.
- **Ollama**: Use the platform’s standard uninstall method. Remove any large model files from Ollama’s data directory if desired.

