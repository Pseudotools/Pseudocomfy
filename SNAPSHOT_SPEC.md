# Pseudorandom Snapshot Specification

This document describes **two successive versions** of the Snapshot (also called a *Spatial Package*) schema used in the Pseudotools / Pseudocomfy ecosystem:

* **Current version (v0.1)** — the structure used by today’s loader and unpacker nodes.  
* **Next version (v0.4)** — a simplified, forward-looking format aligned with the [Workflow Authoring Guide](https://github.com/Pseudotools/pt-essential-workflows/blob/main/AUTHORING_GUIDE.md).

A snapshot captures all per-image data a workflow needs:  
user-defined global prompts, region-specific material prompts and masks, and optional full-frame guidance images (depth, edge, style).

---

## 1. Current Structure (v0.1)

The v0.1 format is what Pseudocomfy currently expects.

### Top-Level Keys

| Key                                 | Type             | Required | Description                                                                 |
| ----------------------------------- | ---------------- | -------- | --------------------------------------------------------------------------- |
| **`pseudorandom_snapshot_version`** | number           | ✅       | Schema version of the snapshot (e.g. `0.1`).                                 |
| **`width`**                         | integer          | ✅       | Target render width in pixels.                                               |
| **`height`**                        | integer          | ✅       | Target render height in pixels.                                              |
| **`pmts_environment`**              | object           | ✅       | Global environment prompts.                                                  |
| **`map_semantic`**                  | array of objects | ✅       | Region-specific material prompts and masks.                                   |
| **`img_depth`**                     | base64 string    | ✅       | Base-64 encoded RGB depth map.                                               |
| **`img_edge`**                      | base64 string    | optional | Base-64 encoded RGB edge or linework pass.                                   |
| **`img_style`**                     | base64 string    | optional | Base-64 encoded RGB global style reference image.                             |

All images are included directly in the JSON as **base-64 PNG or JPEG** strings.

### `pmts_environment`

| Key                | Type   | Required | Description                                      |
| ------------------ | ------ | -------- | ------------------------------------------------ |
| **`pmt_scene`**    | string | ✅       | Scene or composition description.                |
| **`pmt_style`**    | string | ✅       | Global stylistic description (lighting, lens…).  |
| **`pmt_negative`** | string | ✅       | Negative prompt describing elements to avoid.     |

### `map_semantic`

| Key           | Type                  | Required | Description                                                           |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------- |
| **`pmt_txt`** | string or null        | optional | Text prompt for the region’s object or material.                      |
| **`pmt_img`** | base64 string or null | optional | Reference image for the region.                                       |
| **`mask`**    | base64 string         | ✅       | Greyscale mask (must match `width`×`height`).                         |
| **`pct`**     | number                | optional | Optional weighting or coverage factor.                                 |

### Example (v0.1)

```json
{
  "pseudorandom_snapshot_version": 0.1,
  "width": 832,
  "height": 512,
  "pmts_environment": {
    "pmt_scene": "a farm in the grasslands of Iowa at golden hour",
    "pmt_style": "high-quality architectural rendering",
    "pmt_negative": "low-res, watermark, ugly"
  },
  "map_semantic": [
    {
      "pmt_txt": "mid-century modern farmhouse with Shou Sugi Ban siding",
      "pmt_img": null,
      "mask": "<base64 mask>",
      "pct": 0.75
    }
  ],
  "img_depth": "<base64 depth map>",
  "img_edge": null,
  "img_style": null
}
````

---

## 2. Next Structure (v0.4 — Proposed)

The v0.4 format simplifies naming and groups related data into **three clear categories**:

* **`global_guidance`** – user-defined global properties and prompts.
* **`regional_guidance`** – region-specific material prompts and masks.
* **`spatial_guidance`** – full-frame guidance images derived from the 3D model.

### Top-Level Keys

| Key                                 | Type             | Required | Description                                                                      |
| ----------------------------------- | ---------------- | -------- | -------------------------------------------------------------------------------- |
| **`pseudorandom_snapshot_version`** | number           | ✅        | Must be `0.4`.                                                                   |
| **`width`**                         | integer          | ✅        | Target render width in pixels.                                                   |
| **`height`**                        | integer          | ✅        | Target render height in pixels.                                                  |
| **`global_guidance`**               | object           | ✅        | Global user-defined prompts and style image.                                     |
| **`regional_guidance`**             | array of objects | ✅        | Region-specific material prompts and masks.                                      |
| **`spatial_guidance`**              | object           | optional | Model-derived full-frame guidance maps. At least one entry is typically present. |

All images remain base-64 encoded (PNG or JPEG).

---

### `global_guidance`

User-defined properties that describe the entire scene.

| Key                | Type          | Required | Description                                     |
| ------------------ | ------------- | -------- | ----------------------------------------------- |
| **`txt_scene`**    | string        | ✅        | Scene or composition description.               |
| **`txt_style`**    | string        | ✅        | Global stylistic description (lighting, lens…). |
| **`txt_negative`** | string        | ✅        | Negative prompt describing elements to avoid.   |
| **`img_style`**    | base64 string | optional | Full-frame style reference image (base64 RGB).  |

---

### `regional_guidance`

Region-specific prompts describing materials or object details.
Each entry must provide a **mask** and at least one of `txt` or `img`.

| Key        | Type                  | Required | Description                                         |
| ---------- | --------------------- | -------- | --------------------------------------------------- |
| **`txt`**  | string or null        | optional | Region-specific text prompt for material or object. |
| **`img`**  | base64 string or null | optional | Region-specific reference image.                    |
| **`mask`** | base64 string         | ✅        | Region mask image (same `width` × `height`).        |
| **`pct`**  | number                | optional | Optional coverage or weighting factor.              |

---

### `spatial_guidance`

Model-derived full-frame guidance maps that help drive geometry or composition.
Each entry is optional, but **at least one is typically present**.

| Key         | Type          | Required | Description                                  |
| ----------- | ------------- | -------- | -------------------------------------------- |
| **`depth`** | base64 string | optional | Depth map for geometry guidance.             |
| **`edge`**  | base64 string | optional | Edge or linework map for contour guidance.   |
| *(future)*  | base64 string | optional | Additional derived maps such as normals etc. |

---

### Example (v0.4)

```json
{
  "pseudorandom_snapshot_version": 0.4,
  "width": 1600,
  "height": 900,

  "global_guidance": {
    "txt_scene": "Two-story timber atrium with mezzanine ring and clerestory.",
    "txt_style": "Soft daylight, neutral white balance, editorial photo.",
    "txt_negative": "No text, no watermark, no warped structure.",
    "img_style": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
  },

  "regional_guidance": [
    {
      "txt": "white oak planks, matte finish, tight grain",
      "img": null,
      "mask": "data:image/png;base64,iVBORw0KGgoAAA...",
      "pct": 38.2
    },
    {
      "txt": null,
      "img": "data:image/png;base64,iVBORw0KGgoAAA...",
      "mask": "data:image/png;base64,iVBORw0KGgoAAA...",
      "pct": 12.7
    }
  ],

  "spatial_guidance": {
    "depth": "data:image/png;base64,iVBORw0KGgoAAA...",
    "edge":  "data:image/png;base64,iVBORw0KGgoAAA..."
  }
}
```

---

## 3. Summary of Key Changes

| Aspect                 | v0.1 (Current)                                                   | v0.4 (Next)                                                                                  |
| ---------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Global prompts**     | `pmts_environment` with `pmt_scene`, `pmt_style`, `pmt_negative` | `global_guidance` with `txt_scene`, `txt_style`, `txt_negative`, `img_style`                 |
| **Per-region prompts** | `map_semantic` with `pmt_txt`, `pmt_img`, `mask`, `pct`          | `regional_guidance` with `txt`, `img`, `mask`, `pct`                                         |
| **Guidance maps**      | `img_depth` required, `img_edge` optional, `img_style` optional  | `spatial_guidance` with `depth`, `edge`, and any other model-derived maps (e.g., normals)    |
| **Schema version**     | `pseudorandom_snapshot_version`: `0.1`                           | `pseudorandom_snapshot_version`: `0.4`                                                       |
| **Naming**             | Mixed prefixes (`pmt_*`, `img_*`)                                | Clean three-category layout with consistent naming and `_guidance` concept where appropriate |

---

**In brief:**
The **current v0.1** snapshot meets today’s loader requirements.
The **proposed v0.4** snapshot introduces three explicit categories—`global_guidance`, `regional_guidance`, and `spatial_guidance`—to clarify meaning, improve consistency with workflow capabilities, and simplify future expansion.

