# Surveillance alert integration

This package integrates SapiensID identity results, OpenPAR attributes, and
PoseC3D actions without modifying any upstream repository. The implementation
is grouped by responsibility; generated data remains in the MTP root
`input`, `output`, `Database`, and `models` folders.

## Source layout

| Package | Responsibility |
| --- | --- |
| `app/` | Automatic workflow and command-line interfaces |
| `core/` | Data models, paths, JSON I/O, and track association |
| `vision/` | SapiensID identity, OpenPAR mapping, colours, evidence, and PoseC3D action |
| `alerts/` | Email rendering, artifact writing, cooldown, and SMTP delivery |
| `templates/` | Fixed email and person templates |
| `tests/` | Deterministic rendering and safety tests |

The short root modules `main.py`, `auto_workflow.py`, and `cli.py` are stable
compatibility entry points. The implementation lives in the packages above.

## Tracked video workflow

The image workflow remains available. For a prerecorded video, run:

```powershell
& ".\sapiensid\.conda\sapiensid\python.exe" -m src.video `
  ".\input\surveillance.mp4" `
  --location "Main Gate" `
  --source-id "ROBOT-CAM-01"
```

The video workflow samples detections, assigns persistent track IDs, keeps the
three best observations per person, aggregates SapiensID identity evidence,
aggregates OpenPAR attributes, runs PoseC3D on each track clip, and writes a
dry-run alert under `output/video_runs`. It never adds frame detections to
obtain the person count; unique people are counted from track IDs.

Use `--scan-only` to test video detection/tracking without loading SapiensID,
OpenPAR, or PoseC3D. The flags `--skip-identity`, `--skip-attributes`, and
`--skip-action` allow individual stages to be disabled for diagnosis.

Generated files never belong inside `src`; development dry runs are stored in
`output/development_dry_runs`, and automatic runs in
`output/surveillance_alerts`.

## Automatic single-image workflow

From the MTP directory, give the workflow one image:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\RUN_SURVEILLANCE.ps1" `
  -Image ".\input\images.jpg" `
  -Location "Main Gate" `
  -SourceId "CAM-01" `
  -DetectionConfidence 0.30 `
  -MatchThreshold 0.33
```

It automatically detects and crops people, matches each crop against the
root-level identity database with SapiensID, runs OpenPAR on the exact same
crops, binds both outputs by stable track ID, annotates the input, and writes
the fixed email as text, HTML, and EML under
`output/surveillance_alerts/<event-id>`. This remains a safe dry-run and does
not send a live message. If no person is detected, it records that result and
creates no alert email.

Person detection defaults to confidence `0.30`, 640-pixel YOLO inference and
IoU `0.70` non-maximum suppression. Identity matching independently defaults
to similarity `0.33`. In video, total unique people must be calculated
from persistent track IDs rather than summing detections across frames.

The highest-scoring database identity is accepted only when its cosine
similarity is strictly greater than `0.33`. A score equal to or below `0.33`
remains `Unknown`. This rule applies to every identity in the database rather
than being specific to Nicole Kidman.

Garment colours are extracted only from visible pose regions intersected with
the person foreground mask. A colour is omitted when too few pixels are
available or no colour dominates. Carried-item colour is intentionally omitted
until a dedicated human-parsing/item mask is added; the system never assigns a
shirt colour to a bag by guessing.

The controlled colour vocabulary is: `black`, `charcoal`, `grey`, `light grey`,
`white`, `cream`, `beige`, `tan`, `brown`, `maroon`, `red`, `orange`, `gold`,
`yellow`, `olive`, `lime`, `green`, `teal`, `cyan`, `navy`, `blue`, `sky blue`,
`purple`, and `pink`.

Run the relocated SapiensID workflow from the MTP directory with:

```powershell
& ".\sapiensid\.conda\sapiensid\python.exe" -m src.main `
  --image ".\input\images.jpg" `
  --output-dir ".\output" `
  --database-dir ".\Database"
```

## Association contract

The integration never joins people by list position. Supply a tracking manifest
whose entries link one stable track to the exact outputs produced from that
person's crop:

```json
{
  "tracks": [
    {
      "track_id": "17",
      "sapiens_query_label": "person_0",
      "openpar_image_key": "C:\\full\\path\\person_0.jpg",
      "face_evaluated": true,
      "bbox_xyxy": [100, 50, 300, 500],
      "position": "main entrance"
    }
  ]
}
```

For video, `SpatialTemporalTracker` in `core/tracks.py` assigns stable IDs by
bounding-box overlap and frame continuity. Use the returned ID for the crop sent
to both models. A production tracker such as ByteTrack can replace it while
keeping this manifest contract.

## Event metadata

```json
{
  "event_id": "EV-001",
  "event_type": "Person detected",
  "date_time_with_timezone": "2026-11-09 12:00:00 +05:30",
  "location": "Main Gate",
  "source_id": "CAM-01",
  "specific_observed_reason": "Person entered monitored area",
  "first_detected_time": "2026-11-09 12:00:00 +05:30",
  "observed_duration": "10 seconds",
  "status": "Active",
  "snapshot_source": "C:\\path\\frame.jpg",
  "video_clip": "Not available"
}
```

## Dry run (default)

Run from the MTP directory:

```powershell
& ".\sapiensid\.conda\sapiensid\python.exe" -m src.cli `
  --event ".\src\event.json" `
  --sapiens-results ".\sapiensid\output\run\database_match_results.json" `
  --openpar-results ".\OpenPAR\predictions.json" `
  --track-manifest ".\src\track_manifest.json" `
  --output-dir ".\src\dry_runs\EV-001"
```

Dry-run writes plain text, HTML, RFC email (`.eml`), and an annotated snapshot
when bounding boxes and a source snapshot are available. It never sends mail.

Live sending additionally requires `--send` and these environment variables:
`SURVEILLANCE_SMTP_HOST`, `SURVEILLANCE_SMTP_PORT`,
`SURVEILLANCE_SMTP_USERNAME`, `SURVEILLANCE_SMTP_PASSWORD`, and
`SURVEILLANCE_EMAIL_FROM`. The recipient is fixed to
`ayushmohta2402@gmail.com`.

## Supported model information

- SapiensID: database identity and similarity-based known/unknown decision.
- OpenPAR PromptPAR/PETA: supported upper/lower clothing and carried-item labels.
- Activity: `vision/action.py` runs the PoseC3D CPU proof of concept. Low
  confidence remains `performing an undetermined activity`.
- Position: supplied by camera-zone/spatial metadata; otherwise undetermined.
