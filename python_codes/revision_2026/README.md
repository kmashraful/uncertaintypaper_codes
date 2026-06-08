# revision_2026: analysis code for the revised manuscript

Code accompanying the revised version of "Accuracy is not certainty: using model agreement and human judgment to assess spatial uncertainty in high-resolution mangrove mapping" (Environmental Research Letters). These three notebooks reproduce the figures and tables that changed during revision. The original analysis notebooks elsewhere in this repository are unchanged.

## Contents

| Folder | Notebook | Produces |
|--------|----------|----------|
| `figure5_model_vs_interpreter/` | `figure5_model_vs_interpreter.ipynb` | Figure 5 (two-panel: CDF and joint scatter of model vs interpreter standard deviation) |
| `hero_scatter/` | `hero_scatter.ipynb` | Figure 7 (interpreter mean vs stacking-model probability, four panels) |
| `area_estimation_redd/` | `area_estimation_redd.ipynb` | Table 3 (probability bins), Table 4 (area, carbon, REDD+ value), and the supplementary area-estimation methodology |

Each folder has an `outputs/` directory holding the figures, the per-point CSVs, and the result tables.

## What changed in this revision

Two methodological points are worth highlighting for anyone comparing to the original code.

1. Pixel sampling. The model and base-learner probabilities are now read at the centroid pixel at the native 4.77 m PlanetScope resolution, rather than averaged over a 10 m buffer that blurred two to three pixels. This affects Figure 5 and Figure 7. It slightly changed the reported statistics and strengthened the central relationships.

2. Final model and carbon density. The area chain uses the random-forest stacking configuration with feature pass-through. The carbon density is 260 t C per ha, the midpoint of the Sundarbans range in Rahman et al. (2015); the REDD+ price tiers ($5, $15, $50 per tonne CO2e) span the voluntary forest-carbon market (Forest Trends 2024) to compliance-market levels (World Bank 2024).

## Running the notebooks

1. Create an environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Authenticate to Google Earth Engine once:
   ```
   earthengine authenticate
   ```
   The notebooks read interpreter point assets under `projects/ee-islamkm/` and model-prediction assets under `projects/ee-ashrafulcuetbd/` and `projects/ee-islamkm/`. You need read access to those assets, or your own copies repointed in the load cells.
3. Open a notebook and run top to bottom. Each notebook has a header cell describing its inputs, outputs, and asset IDs.

The area-estimation notebook also reads nine local prediction GeoTIFFs. Set `MODELS_DIR` in its config cell to wherever you keep them. They are large and are not included here.

## Verifying the results without Earth Engine or the rasters

The Earth Engine queries and raster reads are the only steps that need privileged access. To make the results checkable without them, each notebook ships its intermediate data in `outputs/`:

- Figure 5 and Figure 7 can be regenerated from `reference_points_with_stats.csv` and `reference_points_with_probs.csv` respectively. Each notebook's header explains the reload path.
- The area estimate and the carbon/REDD table can be recomputed from `bin_counts_final_model.csv` and `reference_points_with_probs.csv`. Set `REPRODUCE_FROM_CSV = True` in the area notebook's config cell and run the final "Quick verification" cell. It reproduces 381,194 ha (95% CI 367,934 to 394,455), 99.1 Mt C, 363.4 Mt CO2e, and the three REDD+ values, with no GEE access.

## License

Released under the same license as the rest of this repository.
