# V6 FINAL BOSS API Documentation

## Base URL
```
http://localhost:5000
```

## Authentication
No authentication required (add JWT for production)

## Endpoints

### GET /
Health check endpoint

**Response:**
```json
{
  "status": "online",
  "model": "V6 FINAL BOSS",
  "version": "6.0 FINAL",
  "architecture": "V4 + DistilBERT Meta-Learning"
}
```

### POST /predict
Single text prediction

**Request:**
```json
{
  "text": "I understand you're going through a difficult time"
}
```

**Response:**
```json
{
  "alignment_score": 0.98,
  "wellbeing_score": 0.85,
  "alignment_label": "high",
  "wellbeing_label": "positive",
  "processing_time_ms": 45.2,
  "model_version": "V6 FINAL BOSS"
}
```

### POST /batch
Batch prediction (max 100 texts)

**Request:**
```json
{
  "texts": [
    "I'm feeling good today",
    "Everything is hopeless",
    "Thank you for your help"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"alignment_score": 0.95, "wellbeing_score": 0.92, ...},
    {"alignment_score": 0.12, "wellbeing_score": 0.08, ...},
    {"alignment_score": 0.88, "wellbeing_score": 0.79, ...}
  ],
  "total": 3,
  "processing_time_ms": 120.5
}
```

### GET /metrics
Model performance metrics

**Response:**
```json
{
  "version": "V6 FINAL BOSS - 50K",
  "accuracies": {
    "alignment": {
      "validation": 100.0,
      "target": 98,
      "achieved": true
    },
    "wellbeing": {
      "validation": 100.0,
      "target": 99,
      "achieved": true
    },
    "overall": 100.0
  }
}
```

## Error Codes
- 400: Bad Request (missing/invalid input)
- 404: Not Found
- 500: Internal Server Error

## Rate Limiting
- 100 requests per minute per IP (configure in production)

## Example Usage

### Python
```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "I appreciate your openness"}
response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Thank you for sharing"}'
```

### JavaScript
```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'I understand your concern'})
})
.then(res => res.json())
.then(data => console.log(data));
```
