# Forest Guardian: segmenting invasive acacia from satellite imagery

🥇 **1st place overall, JunctionX Lisbon 2025** (also 1st in peer vote, 2nd partners' jury, 3rd organisers' jury) · [Técnico write-up](https://tecnico.ulisboa.pt/pt/noticias/campus-e-comunidade/inteligencia-artificial-financas-e-sociedade-tecnico-recebeu-estreia-da-hackaton-junctionx-em-portugal/)

**[▶ Live demo](https://duartepcruz.com/work/invasive-species-mapping/)** (runs in your browser, nothing uploaded) · **[Model](https://huggingface.co/duartepcruz/forest-guardian-yolov8-seg)** · **[Dataset](https://huggingface.co/datasets/duartepcruz/acacia_dataset)** · [Pitch deck (PDF)](docs/Acacia-Classifier-in-Mediterranean-Forests.pdf)

| | | | |
|---|---|---|---|
| ![Acacia stand on a riverbank, outlined](https://duartepcruz.com/figures/acacia-pred-1.webp) | ![Acacia patch in dense forest, outlined](https://duartepcruz.com/figures/acacia-pred-2.webp) | ![Acacia patch near a village, outlined](https://duartepcruz.com/figures/acacia-pred-3.webp) | ![Terraced hillside, nothing flagged](https://duartepcruz.com/figures/acacia-pred-neg.webp) |
| acacia ✅ | acacia ✅ | acacia ✅ | negative, nothing flagged ✅ |

*Validation tiles the model never saw in training.*

Acacias were introduced to Portugal for timber and dune stabilisation, and they now spread through native forest faster than anyone can survey on foot. This project points foresters at where to look.

## Results

YOLOv8n-seg, 15 epochs at 640 px, evaluated on the held-out validation split:

| Metric | Value |
|---|---|
| Box mAP@50 | **0.56** |
| Mask mAP@50 | 0.24 |
| Acacia tiles flagged | 22 / 32 (69%) |
| Negative tiles flagged (false alarms) | **0 / 16** |

The model is deliberately conservative: it would rather miss a stand than send someone into the field for nothing.

**Why masks score much lower than boxes:** the ground truth comes from COS2018 map polygons drawn at landscape scale, not tree crowns. The labels are blobs, so the masks are blobs too. This is a label-quality problem more than a model problem.

### Known failure modes
- **Misses about a third of stands**, especially acacia in yellow bloom. Most training tiles show trees out of flower. Lowering the confidence threshold to 0.15 recovers them, so the signal is there but weak.
- **Masks are coarse** because the labels are coarse (see above). Trust the boxes more than the outlines.
- **One imagery source.** All tiles come from one satellite basemap at a similar zoom. Expect worse results on drone photos, other seasons or other resolutions.

## How the dataset was built

There was no labelled dataset, so we built one during the hackathon ([`backend/core/ml_model_final/data_stuff/`](backend/core/ml_model_final/data_stuff/)):

1. Took acacia polygons from Portugal's official land-use map **COS2018** (class 5.1.1.6) with GeoPandas.
2. Cut square frames around them and pulled matching satellite imagery (Esri World Imagery): `get_images_acacias.py`.
3. Rasterised the same polygons into 640×640 masks with rasterio: `get_masks_acacias.py`.
4. Repeated on non-acacia land to get **negatives**: `get_images_not_acacias.py`, `get_masks_not_acacias.py`.
5. Converted masks to YOLO polygon labels and [published the dataset](https://huggingface.co/datasets/duartepcruz/acacia_dataset): `convert_masks_to_YOLO.py`, `data_to_huggingface.py`.

Training and export: [`backend/core/ml_model_final/ml_training/`](backend/core/ml_model_final/ml_training/) (`training.py`, `export.py`), plus the exploratory notebook in [`ML_training/`](ML_training/).

## System

```
React app ──► FastAPI ──► S3 (images)
                 │
                 └─► Celery + Redis ──► YOLOv8n-seg (Hugging Face) ──► masks + scores ──► Firestore
```

Uploads are queued asynchronously, so large images never block the UI. The checkpoint is also exported to ONNX (13 MB) and runs fully client-side in the [demo](https://duartepcruz.com/work/invasive-species-mapping/), reproducing Ultralytics' prototype-mask decoding in the browser.

![Detection report in the web app](docs/photos/reports-page.png)

**My part:** the dataset pipeline, model training and the whole backend/inference service. My two teammates built the web app.

## Next steps
- Train the full planned 50 epochs (this checkpoint stopped at 15) and compare n/s model sizes.
- Add flowering-season imagery to fix the main recall gap.
- Report per-tile recall at several thresholds, not just the default 0.25.

## Run it locally

Requires Docker, plus an AWS account (S3), a Hugging Face token and a Firebase project.

```bash
./setup.sh           # creates backend/.env, then fill in your credentials
./start_docker.sh    # frontend :3000, API :8000 (docs at /docs), Redis :6379
```

<details><summary>Manual setup, environment variables and troubleshooting</summary>

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
redis-server &          # or: docker run -d -p 6379:6379 redis:7-alpine
./start_backend.sh
```

### Frontend
```bash
cd frontend
npm install
npm start               # http://localhost:3000
```

### Environment variables (`backend/.env`)
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=...
HF_API_TOKEN=...
HF_MODEL_NAME=duartepcruz/forest-guardian-yolov8-seg
FIREBASE_CREDENTIALS=path/to/firebase-service-account.json
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Troubleshooting
- **Health check:** `http://localhost:8000/health`
- **Redis connection error:** make sure `redis-server` is running.
- **Firebase auth error:** check that `FIREBASE_CREDENTIALS` points to a valid service-account file.
- **S3 upload failures:** check AWS credentials and bucket permissions.
- **Model errors:** check the Hugging Face token and model name.
- **Logs:** `docker-compose logs -f`

</details>

## Team
"The Three Musketeers": Duarte Pereira da Cruz, Daniel and André. MIT License.
