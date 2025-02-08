# PersonaGPT

> **This repository is in its final polishing stages.** Feel free to explore the main folder, where all essential files are located.
>
> Stay tuned for updates and improvements!

## Overview

PersonaGPT is an end-to-end model designed to scale from data processing to fine-tuning an entire language model. Its goal is to train a large language model capable of accurately replicating the writing style of the user running the workspace.

The project requires access to a dataset of the user’s personal conversations (see the guide below) for supervised learning. A high-performance GPU, such as Google Colab's A100, is recommended for smooth execution. Initially, PersonaGPT was developed as a personal project to enhance Data Science skills.

## Installation Guide

### Prerequisites

- Ensure Git is installed: [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- A high-performance GPU (e.g., A100) is highly recommended

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/BohdanChuprynka/Stock-Sales-Prediction.git
   ```

2. **Prepare Your Conversation Dataset** PersonaGPT currently supports Telegram and Instagram conversation data for training. Follow these steps to export your data:

   - **Instagram**
     - Navigate to **Download Information** → **Choose Account** → **Some of Your Information** → **Messages ONLY**
     - Download in **JSON** format.
     - More details: [Instagram Data Download Guide](https://help.instagram.com/181231772500920?helpref=faq_content)
   - **Telegram**
     - Go to **Settings** → **Advanced** → **Export Telegram Data**
     - Select **ONLY Personal Chats** → **Machine-readable JSON** → **Export**
     - More details: [Telegram Data Export Guide](https://telegram.org/blog/export-and-more)
   - **Place Your Data in the Project Folder**
     ```
     FilesToRun/
     ├── your_instagram_activity/
     ├── result.json
     ```

3. **Run the Data Processing Script**

   ```bash
   python data_center.py
   ```

## Notes

- Training a model with large datasets requires time and computational power. Please be patient.
- If you encounter issues, feel free to open an issue on GitHub or contact me.

**Enjoy training your own personalized AI! 🚀**

