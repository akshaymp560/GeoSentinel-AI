# GeoSentinel-AI Phase 1

This backend implements the asynchronous investigation API and a deterministic
mock ChangeFormer contract. It does not contain PyTorch, ChangeFormer, GEE, or
any Phase 2+ agents.

## Run

From `backend/`, install `requirements.txt`, optionally copy `.env.example` to
`.env` and set `DATABASE_URL`, then run:

```powershell
uvicorn main:app --reload
```

Swagger is available at `/docs`. If `DATABASE_URL` is not set, the documented
SQLite fallback is used for local development. Set a PostgreSQL URL for shared
or production deployments.

The detector boundary is `services/change_interface.py`; the API only calls
`run_change_detection`, so a future isolated ChangeFormer subprocess can replace
the mock without changing the API layer.

## Checkpoint 2.1: Sentinel-2 retrieval

`services/gee_retriever.py` provides the isolated `retrieve_sentinel2_images`
service. It is not connected to the frozen Phase 1 API yet. Install the
`earthengine-api` dependency, set `GEE_PROJECT`, and authenticate separately
using Google's official Earth Engine workflow; credentials are never stored in
this repository and authentication is not automated by the service.

The service searches `COPERNICUS/S2_SR_HARMONIZED` over a configurable 30-day
window anchored at each requested date, filters cloud cover below 10%, and
selects the least-cloudy image. It writes
`backend/artifacts/investigations/{investigation_id}/before.tif` and
`after.tif`, plus `metadata.json`. Both GeoTIFFs contain exactly these bands in
order: `B2, B3, B4, B8, B11, B12`. Earth Engine is requested to deliver them
on a common 10 m grid using the before image's B2 projection.

The returned dictionary contains `success`, image paths, and acquisition/cloud
metadata; failures contain `success: false`, a safe error message, and a stage
such as `gee_initialization`, `before_query`, `after_query`, or `export`.

## Checkpoint 2.2: spectral analysis

`services/spectral_analysis.py` analyzes the six-band GeoTIFF contract without
using a ChangeFormer mask. It reads bands in the fixed order `B2, B3, B4, B8,
B11, B12`, calculates NDVI, NDWI, and NDBI with safe zero-denominator handling,
and returns deterministic summary statistics. `analyze_before_after` also
returns after-minus-before change summaries. Optional single-band index
GeoTIFFs preserve the input shape, CRS, and transform. Raster processing uses
the `rasterio` dependency.

## Checkpoint 2.3A: Weather evidence agent

`agents/weather_api.py` provides the standalone
`retrieve_weather_evidence` function using Open-Meteo's public archive API.
It returns factual precipitation, temperature, wet-day, and optional wind
metrics for the inclusive requested period. It requires no API key, does not
invent an anomaly baseline, and returns a structured `failed` result for
validation, network, HTTP, or malformed-response errors. It is not connected
to the Phase 1 pipeline or any later evidence orchestrator.

## Checkpoint 2.3B: Fire evidence agent

`agents/fire_api.py` provides the standalone `retrieve_fire_evidence` function
using NASA FIRMS area CSV data (`VIIRS_SNPP_SP`). FIRMS requires a free
`FIRMS_MAP_KEY`, supplied through the environment or function argument; no key
is stored in the repository. The default search area is a 5 km radius converted
to a deterministic latitude/longitude bounding box. Longer periods are queried
in exact chunks of at most five days and are not expanded beyond the requested
dates.

The result reports detection metadata, dates, confidence, FRP, distance, and
counts. Zero detections are a successful observation with `detection_count: 0`.
FIRMS detections represent remotely detected thermal anomalies, not confirmed
wildfires or causal conclusions about satellite change.

## Checkpoint 2.3C: Terrain evidence agent

`agents/terrain_api.py` provides the standalone `retrieve_terrain_evidence`
function using Open-Meteo's public elevation endpoint. It reports point
elevation in metres at the investigation coordinate and does not invent slope
or area statistics because this endpoint is being used for point elevation.
The default terrain context is a configurable 5 km radius, represented as a
deterministic bounding box in the result. No API key is required. Terrain is
returned as contextual evidence only; the agent does not infer landslides,
flooding, causation, or other hypotheses.

## Checkpoint 2.3D: Land Cover evidence agent

`agents/landcover_api.py` provides the standalone
`retrieve_landcover_evidence` function using ESA WorldCover v200 through the
existing Earth Engine dependency. It uses a configurable default 5 km
bounding-box AOI and calculates class percentages from WorldCover pixel
counts. Class names and codes follow the WorldCover definitions, including
tree cover (10), shrubland (20), grassland (30), cropland (40), built-up (50),
bare/sparse vegetation (60), snow and ice (70), permanent water bodies (80),
herbaceous wetland (90), mangroves (95), and moss and lichen (100).

The service requires the existing `GEE_PROJECT` configuration and normal Earth
Engine authentication, but does not automate login or store credentials. Land
cover is contextual evidence only and does not establish deforestation,
development, or causation.

## Sentinel-2 pixel-level preprocessing

`services/gee_retriever.py` preserves the production six-band output
`B2, B3, B4, B8, B11, B12` and additionally exports `before_scl.tif` and
`after_scl.tif` quality sidecars from the Sentinel-2 Scene Classification Layer
(SCL) on the same 10 m CRS/grid. `services/satellite_preprocessing.py` consumes
those sidecars: SCL classes 3 (cloud shadow), 8/9 (cloud), and 10 (cirrus),
plus no-data/defective classes 0/1 and snow/ice class 11, are written as
invalid pixels in aligned six-band outputs. No ChangeFormer-specific
normalization or resizing is performed.

## Checkpoint 2.3E: News evidence agent

`agents/news_api.py` provides the standalone `retrieve_news_evidence` function
using the public GDELT DOC 2.1 article-list API. It applies the requested
inclusive date range, defaults to a maximum of 25 articles (configurable up to
250), and uses GDELT's `near:latitude,longitude,radiuskm` query as a geographic
relevance context. This is a search hint rather than proof that an article is
about the exact coordinate. A custom place or event query can be supplied via
`location_context`. No API key is required. Zero results are successful
evidence collection, and news reports are never treated as causal proof.
