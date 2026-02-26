# CLIP Indie Aesthetics Analysis (Python + HTML/CSS/JS)

This project implements the notebook workflow in a production-friendly form:

- CLIP image embeddings from game screenshot folders
- **3D t-SNE** and **3D UMAP** projections
- Style centroids + cosine similarity matrix
- KMeans clustering + game/cluster cross-tab
- CLIP text-prompt style similarity per game
- Interactive frontend visualizations (Plotly + vanilla JS)

## 1. Install

```bash
cd /Users/gqnsptaa/Desktop/Codex_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional if you specifically want OpenAI's original CLIP package:

```bash
pip install git+https://github.com/openai/CLIP.git
```

The pipeline supports both OpenAI CLIP and OpenCLIP and auto-falls back.

## 2. Dataset Structure (Local Only)

The script now reads screenshots from this fixed folder:

```text
/Users/gqnsptaa/Desktop/Codex_Project/indie_games_dataset/
  GameA/
    img1.jpg
    img2.png
  GameB/
    shot1.webp
```

Use one folder per game. Folder names become labels in the analysis.

## 2.1 AAA vs Indie Group Mapping

Use `src/game_groups.csv` to mark each game as `indie` or `aaa`:

```csv
game,group
Hades,indie
Cyberpunk 2077,aaa
```

The pipeline reads this file automatically (default `--game-groups-file src/game_groups.csv`).
Games missing in this file are assigned to `unassigned`.

## 3. Run Analysis

```bash
python src/clip_indie_pipeline.py \
  --output-dir "web/data" \
  --batch-size 32 \
  --device auto \
  --clip-backend auto
```

To use the graphic-design prompt set from file:

```bash
python src/clip_indie_pipeline.py \
  --output-dir "web/data" \
  --style-prompts-file "src/style_prompts_graphic_design.txt"
```

To use your fine-tuned style adapter checkpoint for the heatmap/scores:

```bash
python src/clip_indie_pipeline.py \
  --output-dir "web/data" \
  --style-adapter-checkpoint "training_outputs/style_adapter/best_style_adapter.pt"
```

Main output:

- `w