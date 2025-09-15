# Pseudocomfy Snapshot Specification

*(Spatial Package JSON – current expected structure)*

A **Snapshot** (also called a *Spatial Package*) captures all of the per-image data a Pseudocomfy workflow needs:
global scene/style prompts, per-object material prompts and masks, and full-frame guidance images such as depth or edge maps.
This document describes the structure that is expected by the current loader and unpacker nodes.

---

## Top-Level Keys

| Key                                 | Type             | Required | Description                                                    |
| ----------------------------------- | ---------------- | -------- | -------------------------------------------------------------- |
| **`pseudorandom_snapshot_version`** | number           | ✅        | Schema version of the snapshot (e.g. `0.1`).                   |
| **`width`**                         | integer          | ✅        | Target render width in pixels.                                 |
| **`height`**                        | integer          | ✅        | Target render height in pixels.                                |
| **`pmts_environment`**              | object           | ✅        | Global environment prompts; see below.                         |
| **`map_semantic`**                  | array of objects | ✅        | List of region-specific material prompts and masks; see below. |
| **`img_depth`**                     | base64 string    | ✅        | Base-64 encoded RGB image representing a full-frame depth map. |
| **`img_edge`**                      | base64 string    | optional | Base-64 encoded RGB image providing an edge or linework pass.  |
| **`img_style`**                     | base64 string    | optional | Base-64 encoded RGB image used as a global style reference.    |

All images are included directly in the JSON as **base-64 encoded strings** (PNG or JPEG).

---

## `pmts_environment`

An object providing global prompts that apply to the entire render:

| Key                | Type   | Required | Description                                                            |
| ------------------ | ------ | -------- | ---------------------------------------------------------------------- |
| **`pmt_scene`**    | string | ✅        | Scene or composition description.                                      |
| **`pmt_style`**    | string | ✅        | Global stylistic description (lighting, lens, medium, etc.).           |
| **`pmt_negative`** | string | ✅        | Negative prompt for elements to avoid (e.g., *no text, no watermark*). |

All three keys are required.

---

## `map_semantic`

An array of objects describing **per-region material prompts**.
Each entry defines a single masked region and may include text and/or image guidance.

| Key           | Type                  | Required | Description                                                                                                     |
| ------------- | --------------------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| **`pmt_txt`** | string or null        | optional | Text prompt describing the object or material in this region. May be null or empty.                             |
| **`pmt_img`** | base64 string or null | optional | Base-64 encoded RGB image giving a visual reference for this region. May be null or empty.                      |
| **`mask`**    | base64 string         | ✅        | Base-64 encoded greyscale image that defines the region mask. Must match the overall `width` × `height`.        |
| **`pct`**     | number                | optional | Intended weight or blend factor (0.0–1.0). Present for forward compatibility; not currently used by the loader. |

Each object in `map_semantic` must include a `mask`.
The number of `pmt_txt`, `pmt_img`, and `mask` entries must be consistent across the array.

---

## Minimal Example

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
      "mask": "<base64-encoded greyscale image>",
      "pct": 0.75
    },
    {
      "pmt_txt": "industrial silo, Lloyd's building by Richard Rogers",
      "pmt_img": null,
      "mask": "<base64-encoded greyscale image>",
      "pct": 0.50
    }
  ],
  "img_depth": "<base64-encoded RGB image>",
  "img_edge": null,
  "img_style": null
}
```

---

## Summary

A Pseudocomfy Snapshot is a single JSON file with:

* **Global prompts** (`pmts_environment`) for scene, style, and negatives.
* **Per-region definitions** (`map_semantic`) combining optional text prompts, optional reference images, and required masks.
* **Full-frame guidance images** (`img_depth` required; `img_edge` and `img_style` optional), all as base-64 encoded images.

This specification reflects the structure required by the current Pseudocomfy loader and unpacker nodes.
