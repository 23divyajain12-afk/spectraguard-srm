# SpectraGuard-SRM UI Reference

## Purpose

Build a beautiful, clean, technically credible, and easy-to-navigate UI
for SpectraGuard-SRM.

The UI is a presentation and interaction layer over the existing
SpectraGuard-SRM pipeline. It must visualize and invoke the existing
pipeline without redesigning or reimplementing the core SR, validation,
FCLS, or data-processing architecture.

The UI should feel like a professional
remote-sensing/scientific-analysis application rather than a generic
admin dashboard.

## Core UI Principles

1.  **Fixed AOI**
    -   The processing AOI is predefined by the current project
        configuration.
    -   The user must NOT draw or redefine the processing AOI.
    -   The fixed AOI should be visibly outlined on the scene map.
    -   The user may pan, zoom, and drag around the displayed scene
        purely for exploration.
    -   Exploring another area must NOT change the processing AOI.
2.  **One expensive SEN2SR inference**
    -   The UI must never trigger a second SEN2SR inference
        unnecessarily.
    -   Standalone SEN2SR output and SpectraGuard-SRM output should
        originate from the existing single inference and branching
        pipeline.
3.  **Pre-caching / presentation-friendly behavior**
    -   The UI should support pre-generated/cached results.
    -   A judge should be able to open the application and immediately
        explore already-processed results without waiting for a long
        SEN2SR inference.
    -   Clearly indicate whether results are cached, available,
        processing, or unavailable.
4.  **No fabricated data**
    -   Never invent metrics, uncertainty values, anomaly values,
        processing times, metadata, or imagery.
    -   If an artifact is unavailable, show a clear "Not available"
        state.
5.  **Normal operation does not require an external HR image**
    -   Internal consistency remains based on degrading the 2.5m
        prediction to the original Sentinel-2 resolution and comparing
        it with the original 10m observation.
    -   External reference evaluation is optional and appears only when
        an external evaluation artifact is available.
6.  **Do not modify core scientific logic**
    -   Do not change SEN2SR inference.
    -   Do not change FCLS mathematics.
    -   Do not change validation methodology.
    -   Do not change spectral-response integration.
    -   Do not change pipeline contracts merely to make the UI easier.

## Desired Navigation

Use a simple left sidebar or similarly clear navigation.

Recommended sections: - Home - Explore Scene - Process - Results -
Validation - About

Primary workflow:

**Explore → Process → Results → Validate**

For presentation mode, cached results should allow the user to go
directly to Results.

# Phase 1 --- UI Foundation

## Goal

Create the application shell and visual design system before
implementing detailed scientific views.

## Requirements

-   Establish the application entry point.
-   Create the main layout.
-   Create sidebar/navigation.
-   Create page/header structure.
-   Create consistent cards, buttons, badges, spacing, typography, and
    status indicators.
-   Add responsive behavior where practical.
-   Include GPU/device status using existing configuration/runtime
    information only.
-   Create empty/loading/error states.

## Design direction

-   Modern dark scientific dashboard.
-   Clean, restrained visual hierarchy.
-   High readability.
-   Avoid excessive gradients, animations, or decoration.
-   Use maps and imagery as the visual focus.
-   Technical details should be available without overwhelming the main
    experience.

Do not implement all pages in this phase.

# Phase 2 --- Explore Scene

## Goal

Create the interactive Sentinel-2 scene exploration page.

Display: - Sentinel-2 scene visualization. - Fixed AOI boundary. - Scene
name/tile information when available. - Acquisition date when
available. - CRS. - Resolution. - Available bands. - Basic scene
metadata.

Interaction: - User can pan/drag the map. - User can zoom in/out. - User
can inspect different visual portions of the scene. - The fixed AOI
remains unchanged. - Clearly label it **Fixed Processing AOI**. - Do NOT
provide drawing tools that change the AOI.

Display controls may include: - Natural color/RGB visualization. -
Available band visualization where practical. - Brightness/contrast
controls if cleanly implementable.

# Phase 3 --- Process / Pre-cached Execution

## Goal

Provide a clean processing view without forcing a long inference during
demonstration.

Show: - Fixed AOI being processed. - Target resolution. - SR model. -
Sampling steps. - Device. - Processing status. - Existing pipeline
progress information where available.

Use the existing pipeline rather than reimplementing processing in the
UI.

Support:

### Cached result available

Show **Results ready** and allow immediate navigation to results.

### No cached result

Show a clear process action and explain that real SEN2SR inference
requires the configured GPU environment.

Never fake progress. If actual pipeline progress is available, display
it.

# Phase 4 --- Results

## Goal

Make the central scientific comparison immediately understandable.

Primary comparison: 1. Sentinel-2 Input --- 10m 2. Bicubic Baseline ---
2.5m 3. Standalone SEN2SR --- 2.5m 4. SpectraGuard-SRM --- 2.5m

Use side-by-side cards, synchronized viewers, tabs, or another clean
comparison mechanism.

Useful interactions: - Zoom/pan. - Band/layer selection where
supported. - Compare views. - Image inspection. - Full-screen image view
if practical.

Do not alter image values merely for presentation unless it is clearly a
visualization-only transformation.

# Phase 5 --- Analysis

Display existing artifacts when available.

### Residual / Detail

-   SR residual/detail information.

### Measurement Consistency

-   Reconstruction/degradation consistency information.
-   Relevant existing metrics.

### Uncertainty

-   Uncertainty map.
-   Clear legend.
-   Explain that uncertainty is an analytical confidence/uncertainty
    output, not an external HR-reference score.

### Anomaly

-   Anomaly map/mask if available.
-   Clear normal/anomaly legend.

### Spectral Analysis

Provide a spectral comparison visualization where existing data supports
it.

Possible curves: - Original/degraded Sentinel-2 observation. -
Bicubic. - Standalone SEN2SR. - SpectraGuard-SRM.

Do not fabricate curves.

# Phase 6 --- Validation

Clearly separate internal operational validation from optional external
evaluation.

## Internal validation

Show:

**2.5m result → degrade to 10m → compare against original Sentinel-2
observation**

Make this distinction clear.

## External evaluation

Only show when an external evaluation artifact exists.

Examples: - MuS2 evaluation. - Other controlled benchmark/reference
evaluation.

Clearly label this **External Evaluation** and never imply that external
HR imagery is required for normal operation.

Metrics may include only values actually present: - PSNR - SSIM - SAM -
RMSE - MAE - ERGAS

# Phase 7 --- FCLS / Endmember Information

Show when available: - Endmember library/source. - Material
names/classes. - Abundance information. - FCLS constraints. - Residual
information.

Important: - SEN2SR uses B02, B03, B04, B08. - B11 must never be
fabricated for SEN2SR. - The original five-band USGS endmember set may
exist internally, while the SEN2SR/FCLS operational view is four-band. -
Do not change FCLS mathematics.

Place this under an expandable **Technical Analysis** area if needed.

# Phase 8 --- Metadata / About

Show existing metadata such as: - Project name/version. - Scene. -
CRS. - Input resolution. - Output resolution. - Bands. - SR
model/version. - Sampling steps. - Device. - AOI size. - Processing
timestamp where available. - Artifact/provenance information.

Do not invent missing metadata.

Include a concise explanation:

**What is SpectraGuard-SRM?**

SpectraGuard-SRM uses a swappable super-resolution model and adds
spectral-spatial analysis, measurement/degradation consistency,
uncertainty/anomaly analysis, and endmember-based interpretation around
the SR output.

Do not describe SEN2SR as if it were the entire contribution.

# Presentation Mode

The intended live-demo flow:

1.  Open application.
2.  Show the Sentinel-2 scene.
3.  Pan/zoom briefly to demonstrate scene exploration.
4.  Show the fixed processing AOI.
5.  Open pre-cached processing/results.
6.  Show Sentinel-2, Bicubic, standalone SEN2SR, and SpectraGuard-SRM.
7.  Show uncertainty/anomaly/measurement-consistency analysis.
8.  Show validation.
9.  Show technical/FCLS information only if useful.

The judge should NOT need to draw an AOI or wait for a fresh SEN2SR
inference.

# Implementation Constraints

-   First inspect the existing repository and determine the existing
    UI/web stack.
-   Reuse existing dependencies whenever possible.
-   Prefer a Python-native dashboard approach if consistent with the
    project.
-   Keep UI code modular.
-   Keep UI separate from core processing logic.
-   Do not duplicate pipeline logic inside the UI.
-   Do not add a second SEN2SR execution path.
-   Do not change scientific algorithms.
-   Do not change core schemas unless absolutely necessary and
    explicitly justified.
-   Avoid large new dependencies.
-   Do not run real SEN2SR during UI development.
-   Use fixture/cached artifacts for UI development when possible.

# Testing Strategy

UI work is intentionally phased.

After each phase: 1. Run only focused checks/tests relevant to that
phase. 2. Inspect the UI manually when possible. 3. Do NOT run the
entire test suite after every small UI change.

After all UI phases are complete: - Run the full automated test suite
once. - Then perform real GPU/integration testing separately.

# Definition of Done

The UI is complete when: - The application launches reliably. -
Navigation is intuitive. - Sentinel-2 imagery can be explored by
pan/zoom. - The processing AOI remains fixed. - Cached results can be
viewed without rerunning expensive inference. - The four primary result
views are clearly compared. - Uncertainty/anomaly/consistency outputs
are displayed when available. - Internal and external validation are
clearly separated. - FCLS/endmember information is available without
overwhelming the main workflow. - Missing artifacts are handled
honestly. - The UI does not modify the scientific pipeline
architecture. - The UI is suitable for a technical project
demonstration.
