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

To use the expanded 108-prompt set while keeping UI heatmaps readable:

```bash
python src/clip_indie_pipeline.py \
  --output-dir "web/data" \
  --style-prompts-file "src/style_prompts_graphic_design_expanded.txt" \
  --prompt-focus-file "src/style_prompts_graphic_design_focus.txt"
```

`--prompt-focus-file` filters only the displayed/exported heatmap prompt matrices
to a curated subset (default: `src/style_prompts_graphic_design_focus.txt`).
The full prompt set is still scored and exported to `*_full.csv` + JSON `*_full` sections.

To use your fine-tuned style adapter checkpoint for the heatmap/scores:

```bash
python src/clip_indie_pipeline.py \
  --output-dir "web/data" \
  --style-adapter-checkpoint "training_outputs/style_adapter/best_style_adapter.pt"
```

Main output:

- `web/data/analysis_results.json`
- `web/data/sample_points.csv`
- `web/data/centroid_similarity.csv`
- `web/data/prompt_similarity_by_group.csv`
- `web/data/prompt_similarity_by_game.csv`
- `web/data/prompt_similarity_by_group_full.csv`
- `web/data/prompt_similarity_by_game_full.csv`
- `web/data/cluster_crosstab.csv`

How CLIP uses prompts:
- Images are encoded into CLIP embeddings.
- Prompts are encoded into text embeddings.
- Similarity is computed as cosine similarity (`image_embedding dot prompt_embedding`).
- Scores are averaged by game and exported to `prompt_similarity_by_game.csv`.

## 4. Launch UI (with Run Analysis button)

Start the integrated local app server:

```bash
python src/local_app_server.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

If `training_outputs/style_adapter/best_style_adapter.pt` exists, the Run Analysis button
automatically passes it to the pipeline.
If `src/style_prompts_graphic_design_expanded.txt` and
`src/style_prompts_graphic_design_focus.txt` exist, Run Analysis also uses them automatically.

From the page you can now click:
- `Run Analysis` to execute the full Python pipeline.
- `Load Default Data` to reload `web/data/analysis_results.json`.
- Adjust `UMAP n_neighbors`, `UMAP min_dist`, and `t-SNE perplexity` sliders, then click `Run Analysis` again to compare structures.
- Explore the new `Neighborhood Radius Explorer` and `2D UMAP/t-SNE image maps` sections.
- Use `Group Filter` and group-level heatmaps to compare `aaa` vs `indie`.

## 5. Alternative: Static Frontend only

```bash
cd web
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## 6. Train a Style Adapter (OpenCLIP)

Create a labeled CSV with columns `path,label`:

```text
path,label
indie_games_dataset/Hades/shot1.jpg,cinematic low-key lighting
indie_games_dataset/Fez/img3.png,flat geometric vector design
```

Train an adapter head on frozen OpenCLIP embeddings:

```bash
python src/train_openclip_style_adapter.py \
  --train-csv path/to/style_labels.csv \
  --image-root /Users/gqnsptaa/Desktop/Codex_Project \
  --output-dir training_outputs/style_adapter \
  --model-name ViT-B/32 \
  --pretrained openai \
  --device auto \
  --epochs 30 \
  --batch-size 64 \
  --cache-embeddings
```

Outputs:
- `training_outputs/style_adapter/best_style_adapter.pt`
- `training_outputs/style_adapter/training_summary.json`
- Optional embedding caches when `--cache-embeddings` is used.

After training:
- Run analysis with `--style-adapter-checkpoint ...` (or just click `Run Analysis` in UI if the default checkpoint path exists).
- The `Prompt Similarity by Game` heatmap will switch to adapter-based style scores.

## 7. Thesis Attribute Analysis (Indie vs AAA)

Run the thesis-focused analysis script:

```bash
python src/thesis_attribute_analysis.py \
  --output-dir web/data/thesis \
  --device auto \
  --clip-backend auto \
  --model-name ViT-B/32
```

This script extracts and compares:
- `color_*` features (saturation, luminance contrast, warm/cool usage, palette entropy)
- `composition_*` features (center bias, rule-of-thirds energy, left-right symmetry, negative space)
- `texture_*` features (entropy, Laplacian variance, local contrast, high-frequency ratio)
- `typography_*` proxies (edge density, horizontal-vs-vertical stroke ratio, text-like edge density)
- `affect_*` CLIP prompt scores (default affective prompts or custom list)

By default it aggregates features at **game level** (`--use-game-aggregation`) and trains a
balanced logistic classifier for `indie` vs `aaa`.

Outputs:
- `web/data/thesis/attribute_features_per_image.csv`
- `web/data/thesis/attribute_features_modeling_table.csv`
- `web/data/thesis/attribute_feature_group_means.csv`
- `web/data/thesis/attribute_feature_importance.csv`
- `web/data/thesis/attribute_feature_stats.csv`
- `web/data/thesis/attribute_analysis_report.json`

Optional custom affective prompts:

```bash
python src/thesis_attribute_analysis.py \
  --affective-prompts-file src/affective_prompts_thesis.txt
```

## Notes on Robustness and Performance

- Corrupted/unreadable images are skipped and listed in `skipped_images` output.
- Small datasets automatically use PCA fallback where t-SNE/UMAP are unstable.
- t-SNE perplexity and UMAP neighbors are auto-clamped to valid ranges.
- Batched CLIP inference is used for speed, with CUDA mixed precision where available.
- Thumbnails are exported to `web/data/thumbs/` for image-marker visualizations.
- Frontend uses `Plotly.react` and filtered rendering to keep interactions responsive.
