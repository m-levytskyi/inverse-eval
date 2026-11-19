# Project review and focused polishing plan

Preserved from the review of 2026-10-29 at commit `fc95407391640443ae0af00ce6ecc61ad7079737`.
This records the review and agreed implementation scope, not the current completion status.

## Summary

This repository evaluates reflectometry models, compares predictions with known structures, and produces batch statistics and plots. The review assumes a researcher running individual experiments and sequential batches containing hundreds or thousands of curves.

The main problems are scientific assumptions and interfaces that have drifted apart, rather than formatting. Several workflows can produce misleading results despite passing tests.

Review baseline: 124 tests passed; Ruff and `pip check` passed; the configured type check failed. Small reproductions confirmed the findings without running inference. Source files and existing artifacts were unchanged during the review.

## Findings

### Must fix

1. **Nonzero ambient SLD is discarded.** The parser retains fronting roughness but drops fronting SLD; neither inference path forwards it. In the inspected test dataset, 296 of 374 models passing current constraints have nonzero fronting SLD. Preserve it through inference and SLD plotting, with bounds validated in the model's shifted coordinates. Otherwise these curves are evaluated with the wrong ambient assumption. Evidence: `parameter_discovery.py:218`.

2. **Requested prior percentages do not describe the effective NF priors.** Requests for 30%, 50%, and 99% all produce SLD widths of 5, although the NF model supports 24. That cap belongs to the point estimator. Derive inference limits from the selected model and save effective bounds; preserve the existing constraint-MAPE denominator. Otherwise sweep labels overstate how much the priors changed. Evidence: `parameter_discovery.py:470`.

3. **File discovery can select the wrong layer or combine unrelated files.** A layer-1 request accepts files under `2/`; recursion can also pair a curve in one directory with a model in another. Select matching pairs within one directory, validate layer count, sort deterministically, and reject ambiguous duplicates. Keep support for flat datasets. Evidence: `parameter_discovery.py:79`.

4. **Pickle-to-file matching silently breaks for subsets.** The denoising and comparison utilities associate rows with the current file-list position. Removing an earlier file shifts every subsequent pairing. For positional arrays, validate the complete documented ID sequence before processing; reject missing or filtered inputs instead of silently associating the wrong curves. Evidence: `build_denoised_experimental_dataset.py:215`.

5. **NF preprocessing applies a second, different error filter.** Root settings can preserve a point that the vendor subsequently removes using its own 30% threshold. This also undermines "preprocessing disabled." Let the root own configurable error filtering and the vendor perform NF interpolation once; explicitly disable the vendor's additional error filter. Evidence: `simple_pipeline.py:358`.

6. **Invalid posterior rows can win production selection.** With likelihoods `[0, NaN, -1]`, production selects the NaN row while the diagnostic selects the valid maximum. Share finite-row validation and global-maximum selection, preserve original row indices, and reject inconsistent array shapes or entirely invalid output. Otherwise production and "current ML" diagnostics disagree. Evidence: `simple_pipeline.py:175`.

7. **Sweeps can run a different workload from the requested one.** Sweep commands omit `disable_preprocessing`, `disable_constraints`, and explicit experiment IDs. Prominent-feature selection separately replaces the requested experiment limit with the entire filtered count. Forward all supported settings and apply the limit after filtering. These bugs can both change results and unexpectedly increase runtime. Evidence: `batch_sweep_runner.py:107`, `batch_pipeline.py:288`.

8. **Batch failures can appear successful, and interruptions lose completed work.** Results are saved only after the processing loop. An interruption, or the sweep's two-hour timeout, can discard completed experiments. An all-failed batch also exits successfully. Persist completed records incrementally and propagate meaningful completion status through batch and sweep commands. Evidence: `batch_pipeline.py:606,794`.

9. **Model comparison cannot consume current summaries.** It requires `summary["debug_info"]`, which the current writer no longer produces. A current successful batch therefore yields no comparison data. Read metrics from the existing per-experiment `batch_results.json`; do not restore a duplicate summary payload. Evidence: `compare_model_versions.py:95`.

10. **The random baseline breaks on normal NF outputs.** It uses prediction names containing nuisance parameters such as `r_scale`, while truths and priors contain only physical parameters. A standard seven-parameter NF record raises an unknown-constraint error. Use canonical truth names and shared metrics, and append model/random scores as matched pairs. Also distinguish truth-conditioned random priors from an independent uniform baseline. Evidence: `evaluate_random_guessing.py:177`.

11. **Aggregation can report contradictory sample counts.** Two batches with the same setup overwrite one another in the setup dictionary, while both contribute metric arrays. A reproduced result reported one success but two metric observations. Reject duplicate setup labels and ambiguous batch-directory matches under the current one-batch-per-setup contract. Evidence: `aggregate_batch_stats.py:221`.

12. **Plots can overwrite earlier configurations or omit results.** Comparison code constructs unique filenames but passes only their parent directory; helpers then save constant filenames. Two-layer parameter plots also stop after six of eight parameters. Give each comparison its own output directory, size grids from parameter count, and show an overflow bin for errors at or above 100%. Evidence: `compare_model_versions.py:274`, `plotting_utils.py:656`.

13. **Notebook and documentation contracts are stale.** Notebook 05 passes removed keywords to the metric function and raises `TypeError` after inference. README examples use the wrong default data location, and the README/notebook explain constraint-MAPE as division by prior half-width rather than fixed global widths. Repair the calls, examples, and scientific explanation together. Evidence: `notebooks/05_multimodal_posterior_mape.ipynb:217`, `README.md:477`.

14. **One advertised CUDA installation cannot succeed.** The project pins `torch==2.7.1+cu121`, but PyTorch 2.7.1 provides CUDA 11.8, 12.6, and 12.8 builds. Remove the unavailable backend and let automatic selection choose `cu118` for the affected driver range. Evidence: `requirements.torch-cu121.txt:3`; https://pytorch.org/get-started/previous-versions/#v2-7-1.

### Should fix

15. **Noise policy and posterior-summary labels hide important assumptions.** The sigma-input flag also changes likelihood noise. Importance weights are calculated upstream, but stored means and percentiles remain unweighted. Following the user's choices, preserve existing numerical defaults, record the effective noise policy, label these as unweighted proposal statistics, and retain effective sample size. Evidence: `simple_pipeline.py:375`.

16. **A root commit does not identify the inference environment.** Setup pulls the moving `dev_ml` framework branch, dependencies float, and outputs omit important model/configuration identities. Pin the inspected framework revision and save resolved run configuration, framework SHA, package versions, model identity, and effective priors. Existing results should remain readable with provenance marked unavailable. Evidence: `bootstrap.py:25`.

17. **Quality commands modify the environment they check.** Make targets install dependencies and can replace Torch before testing. Separately, ty cannot resolve the editable framework; fixing its search path exposes an invalid `None` receiver in the ambient-SLD test. Make checks use the existing environment, add the vendor search path, and instantiate the test object normally. Evidence: `Makefile:103`, `tests/test_inference_ambient_sld.py:11`.

18. **There is concrete redundancy worth removing.** Comparison metadata is read and passed into plotting arguments that never use it; notebook/script helpers contain exact duplicates; each batch experiment constructs and loads the same model again. Remove unused metadata plumbing, share the duplicated helpers, and reuse one model within a homogeneous batch. Approximately 70 duplicated or ineffective lines were identifiable before broader refactoring. Evidence: `compare_model_versions.py:257`, `simple_pipeline.py:507`.

## Implementation sequence

### 1. Repair scientific input handling and selection

- Store ambient SLD alongside each parsed parameter block; add optional `ambient_sld=0.0` to inference wrappers.
- Pass canonical model ranges and width limits into prior generation. Requested widths remain percentages of the existing evaluation ranges.
- Intersect inference support with the evaluation domain after accounting for ambient shift. Explicitly exclude unsupported truths.
- Use model-supported minimum widths for approximately fixed SLDs.
- Consolidate finite sample selection and remove duplicate NF filtering/interpolation. For a singleton sample, store unavailable standard deviations as `null`.

### 2. Make batch execution trustworthy

- Correct option forwarding, deterministic discovery, and experiment limits.
- Save each completed record atomically in the run directory; retain the existing final batch JSON format.
- Return failure for actual processing errors, propagate interruption as exit 130, and report all-outlier runs explicitly as having no eligible results.
- Let sweeps continue collecting outcomes but return failure when any child fails.

### 3. Repair consumers and remove proven duplication

- Make comparison, baseline, aggregation, and replotting consume the same current result records.
- Preserve model/random pairing, reject duplicate aggregation inputs, and retain every plot.
- Repair notebook calls and documentation; reuse existing modules for shared helpers.
- Remove unused comparison metadata plumbing. Treat the two locally unused convenience wrappers as deletion candidates only; external notebook usage is unverified.
- Reuse a model per batch through an optional injected model argument, without introducing a global cache.

### 4. Stabilize setup and provenance

- Correct CUDA selection and make checks independent of installation.
- Fix ty resolution and its exposed test error.
- Pin framework revision `04427478bfc6e4ad057eb3449df0dffa24253bc8`.
- Save run metadata once per output directory and link sweep outcomes to child directories. Defer packaging changes and a cross-platform dependency-lock migration.

## Tests and acceptance criteria

Add regression tests for zero/nonzero ambient, shifted bounds, both model width limits, boundary truths, 5/30/99% priors; wrong-layer discovery, ambiguous pairs, missing positional rows, deterministic selection; invalid/mismatched/singleton posterior output; sweep forwarding and limits; mixed/all-failed/all-excluded outcomes and interruption after one completed experiment; current writer-to-reader integration, duplicate aggregation inputs, distinct plots and eight-parameter grids; notebook helper calls, documented parser examples, CUDA selection, and checks that perform no installation.

Acceptance requires the complete test suite, Ruff, configured ty checks, and `git diff --check` to pass. Scientific changes need a separately approved, explicitly configured inference comparison before interpreting differences in historical results.

## Decisions and limits

Preserve global `argmax(log_likelihoods)` constraint-MAPE over valid samples, fixed normalization widths, and the separate oracle diagnostic. Preserve the default NF noise model `0.2 * R + 1e-12` and existing unweighted summary values. The user chose focused cleanup over structural cleanup or release-readiness work, preserving the noise default and labeling existing summaries instead of adding weighted statistics.

GitNexus reported CRITICAL impact for prior generation and HIGH impact for the single-run workflow. Update callers together, repeat impact analysis before edits, and require complete graph-change analysis before commits.

Full inference, training, complete notebook execution, GPU/MPS/Windows behavior, and the entire vendored framework were not checked. The main unresolved risk is how much corrected scientific behavior changes historical conclusions.
