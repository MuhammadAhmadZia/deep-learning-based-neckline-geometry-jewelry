# Neckline Geometry Measurement and Rule-Based Jewelry Pairing

Replication package for a study that measures neckline geometry directly from a
single catalog photograph and uses that measurement, rather than the garment's
catalog label, to choose a jewelry pairing.

Everything reported in the paper can be recomputed from the tables in `results/`
by running one script. No GPU and no image downloads are needed for that step.

## Authors

Munaza Munir<sup>1</sup>, Aima Asim<sup>1</sup>, Muhammad Ahmad Zia<sup>1</sup>,
Atif Ikram<sup>1,2</sup>, Asef Mohammad Ali Al Khateeb<sup>3</sup>,
Marwan Abu-Zanona<sup>4,\*</sup>, Najla Abdulaziz Almousa<sup>4</sup>,
Hebah Abdullah Abubakr<sup>4</sup>, Abdulmalik AlJabr<sup>5</sup>

1. Department of Computer Science & IT, The University of Lahore, Lahore 54600, Pakistan
2. Faculty of Computer Science and Mathematics, Universiti Malaysia Terengganu, Kuala Terengganu, Malaysia
3. Department of Computer Science, Faculty of Computer Science and Information Technology, Jerash University, Jerash 26150, Jordan
4. Department of Management Information Systems, College of Business Administration, King Faisal University, Al-Ahsa 31982, Saudi Arabia
5. Applied College, King Faisal University, Al-Ahsa 31982, Saudi Arabia

\* Corresponding author: Marwan Abu-Zanona (mabozanoneh@kfu.edu.sa)

**Repository maintainer:** Muhammad Ahmad Zia
([muhammad.zia@cs.uol.edu.pk](mailto:muhammad.zia@cs.uol.edu.pk),
[ahmadzia.com](https://ahmadzia.com),
ORCID [0000-0002-3208-6246](https://orcid.org/0000-0002-3208-6246)).
Questions about the code, the result tables or the verification script go to the
maintainer. Questions about the study itself go to the corresponding author.


## Quick start

```bash
pip install -r requirements.txt
python scripts/verify_results.py
```

To redraw the figures, run the figure scripts from inside `scripts/`, since they
resolve `../results/` and `../figures/` relative to themselves:

```bash
cd scripts && python fig3_openness_by_type.py
```

The script recomputes the detector summary, the geometry table, the separation
test, the rule assignment, the blind control study and the paired comparison,
and prints each value next to the one reported in the paper.

## What is in here

| Path | Contents |
|---|---|
| `results/recommendations_final.csv` | One row per admissible detection: neckline type, box dimensions, openness ratio, detector confidence, assigned category, recommended necklace and earrings, and whether the image was used in detector training |
| `results/geometry_table_final.csv` | Median and interquartile range of the openness ratio per neckline type, unseen images only |
| `results/seed_comparison.csv` | Test-set detector metrics for seeds 0, 1 and 2 |
| `results/human_eval_control_study.csv` | The 150-image rating round, including the 30 silently swapped controls and all three rater responses |
| `results/type_vs_geometry_paired.csv` | The 30 matched pairs comparing per-image geometry against the label-only baseline |
| `notebooks/neckline_geometry_pipeline.ipynb` | The full development notebook: dataset conversion, detector training, inference, geometry extraction, duplicate audit and every analysis |
| `scripts/verify_results.py` | Recomputes all reported statistics from `results/` |
| `scripts/fig*.py` | Figure code, run from inside `scripts/`. `figlib.py` holds the shared palette and drawing helpers |
| `figures/` | The figures as published |

## Reproducing from scratch

The notebook runs end to end on the free tier of Google Colab with a T4. It
expects Google Drive to be mounted and writes every intermediate result there,
so a disconnected session can be resumed without losing work.

Order of operations:

1. Convert Fashionpedia and the Kouture kurta set into YOLO format and build the
   merged corpus, split 80 / 10 / 10 by product identifier under a fixed seed.
2. Train YOLOv8n at 640 px for 30 epochs. Three seeds were trained; the seed-0
   checkpoint is used for everything downstream.
3. Run inference over the kurta images and record the neckline box.
4. Compute the openness ratio as box width divided by box height in pixel space,
   then apply the box-area plausibility filter.
5. Fit the quantile cut points on open necklines only and assign categories.
6. Run the evaluation arms.

Detector checkpoints are not included because of their size. Training the three
seeds takes roughly three hours per seed on a free Colab T4.

## Data

Two public sources are used. Fashionpedia supplies the labeled neckline regions
that teach the detector where a neckline is. The Kouture kurta set, obtained from
Kaggle under an MIT license, supplies South Asian garments with neckline type
labels; its creators describe the images as sourced from various web platforms.
The annotation file accompanying it matches the format of a Roboflow export, and
the original annotator could not be confirmed.

Images are not redistributed here. The result tables key every row to a product
identifier so the analysis can be audited without the images.

## Notes on the analysis

The openness ratio is taken in pixel space as box width over box height. Dividing
each dimension by a different image dimension would let the photograph's own
aspect ratio enter the measurement, so it is not done that way.

Admissibility filtering operates on box area as a fraction of the image. It never
filters on the openness ratio itself, because conditioning on the outcome
variable would bias every statistic computed from it.

Closed collars are routed by type rather than by geometry. A single scalar ratio
cannot separate a narrow open neckline from a closed band collar, since both
produce a tall narrow box, and the paper reports this as a limitation of the
measure.

The comparison between per-image geometry and the modal category of each neckline
type is circular when both sides are derived from the same variable. The
non-circular evidence is the paired human comparison in
`results/type_vs_geometry_paired.csv`.

## License

MIT, see `LICENSE`. The datasets carry their own licenses.

## Citation

Citation metadata is in `CITATION.cff`. Zenodo deposit metadata is in
`.zenodo.json`; commit it to the repository root before tagging a release so the
archived record carries the right authors.
