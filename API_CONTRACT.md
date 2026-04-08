# API Contract — EpiGraph-AI

Agreed interface between backend (Person A) and frontend (Person B).
**Do not change these response shapes without both people agreeing.**

---

## `GET /api/districts`
```json
{
  "districts": [
    { "name": "Ahmedabad", "color": "#3b82f6" },
    { "name": "Gandhinagar", "color": "#8b5cf6" },
    { "name": "Rajkot", "color": "#06b6d4" },
    { "name": "Surat", "color": "#ec4899" },
    { "name": "Vadodara", "color": "#10b981" }
  ]
}
```

## `GET /api/predictions`
```json
{
  "Ahmedabad":   { "risk": 20.2, "level": "high" },
  "Gandhinagar": { "risk": 19.9, "level": "high" },
  "Rajkot":      { "risk": 8.5,  "level": "low" },
  "Surat":       { "risk": 17.6, "level": "medium" },
  "Vadodara":    { "risk": 27.3, "level": "high" }
}
```

Risk levels: `"high"` (risk > 15), `"medium"` (risk 10-15), `"low"` (risk < 10)

## `GET /api/cases?district=Ahmedabad`
```json
{
  "district": "Ahmedabad",
  "cases": [
    { "date": "2010-01-03", "value": 24 },
    { "date": "2010-01-10", "value": 251 }
  ]
}
```
Returns all weekly case records for the district. If `district` param is omitted, returns all districts.

## `GET /api/weather?district=Ahmedabad`
```json
{
  "district": "Ahmedabad",
  "tmax": 34.2,
  "tmin": 21.5,
  "rain": 52.3,
  "rh_am": 68,
  "rh_pm": 42
}
```
Returns average weather over the full time range.

## `GET /api/news?district=Ahmedabad`
```json
{
  "district": "Ahmedabad",
  "alerts": [
    { "date": "2010-01-03", "headline": "Health Emergency: Ahmedabad hospitals report surge in viral fever.", "type": "Medical_Alert" },
    { "date": "2010-01-17", "headline": "Alert: Heavy monsoon and waterlogging reported in Ahmedabad.", "type": "Weather_Alert" }
  ]
}
```
Types: `"Medical_Alert"`, `"Weather_Alert"`

## `GET /api/connectivity`
```json
{
  "edges": [
    { "source": "Ahmedabad", "target": "Gandhinagar", "weight": 1.0 },
    { "source": "Ahmedabad", "target": "Vadodara", "weight": 0.9 },
    { "source": "Ahmedabad", "target": "Surat", "weight": 0.7 },
    { "source": "Surat", "target": "Vadodara", "weight": 0.8 },
    { "source": "Rajkot", "target": "Ahmedabad", "weight": 0.5 },
    { "source": "Rajkot", "target": "Surat", "weight": 0.4 }
  ]
}
```
