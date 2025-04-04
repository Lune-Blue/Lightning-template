# ⚡ PyTorch Lightning Training Template

A modular and extensible deep learning training template built with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).  
This project is designed to support flexible and scalable training of transformer-based models with built-in support for:

- ✅ HuggingFace Transformers
- ✅ PEFT / LoRA Fine-tuning
- ✅ Accelerate-based training
- ✅ Custom training configs via YAML
- ✅ W&B logging
- ✅ Learning rate finder
- ✅ Continual training & testing

---

## 🚀 Project Structure

```bash
.
├── main.py                    # Main training entry point
├── config/                   # YAML config files
├── model/
│   ├── BackboneModel.py      # Lightning model definition
│   └── trainer.py            # LightningModule wrapper
├── data/
│   ├── dataclass.py          # Argument schema definitions
│   ├── dataset.py            # DataModule definition
├── utils/
│   ├── utility.py            # Helpers (e.g., accelerate model, checkpointing)

🧩 Config-based training
Easily customize all arguments (model, data, training, LoRA, generation, etc.) using a YAML config.

⚡ Lightning Trainer with Callbacks
Integrated with ModelCheckpoint, EarlyStopping, LearningRateMonitor, and custom LR finder.

📚 LoRA / Adapter Support
Lightweight fine-tuning with PEFT for large language models.

🧪 Testing Mode
Automatically loads best checkpoints and evaluates model.

📊 Weights & Biases Logging
Full integration with wandb for tracking metrics and artifacts.