# FeatureForge API Reference

## Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check API health |

## Datasets

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/datasets/upload` | Upload and register CSV dataset |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/{dataset_id}` | Get dataset details |
| GET | `/api/datasets/{dataset_id}/preview` | Preview dataset rows |

## Features

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/features` | Create feature definition |
| POST | `/api/features/validate` | Validate feature definition |
| GET | `/api/features` | List feature definitions |
| GET | `/api/features/{feature_id}` | Get feature definition |
| GET | `/api/features/{feature_id}/preview` | Preview computed feature |

## Materializations

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/materializations` | Materialize features into offline table |
| GET | `/api/materializations` | List materializations |
| GET | `/api/materializations/{materialization_id}` | Get materialization metadata |
| GET | `/api/materializations/{materialization_id}/preview` | Preview materialized table |

## Models

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/models/train` | Train model from materialization |
| GET | `/api/models` | List trained models |
| GET | `/api/models/{model_id}` | Get model metadata |
| GET | `/api/models/{model_id}/metrics` | Get model metrics |

## Predictions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predictions/models/{model_id}` | Predict from JSON records |
| POST | `/api/predictions/models/{model_id}/batch` | Predict from materialized table |
| GET | `/api/predictions/models/{model_id}/input-schema` | Get expected feature columns |

## Online Store

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/online-store/materialize` | Push materialization to online store |
| GET | `/api/online-store/{materialization_id}/features/{entity_value}` | Lookup online feature vector |
| POST | `/api/online-store/{materialization_id}/batch-lookup` | Batch online feature lookup |
| GET | `/api/online-store/{materialization_id}/stats` | Online store stats |
| POST | `/api/online-store/models/{model_id}/predict` | Predict using online features |

## Drift Monitoring

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/drift/reports` | Generate drift report |
| GET | `/api/drift/reports` | List drift reports |
| GET | `/api/drift/reports/{report_id}` | Get drift report |
