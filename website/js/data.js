// EpiGraph-AI — District metadata for maps (D3 handles actual geo-rendering)

// District display colors and metadata
const DISTRICT_META = {
    Ahmedabad:   { color: "#3b82f6" },
    Gandhinagar: { color: "#8b5cf6" },
    Rajkot:      { color: "#06b6d4" },
    Surat:       { color: "#ec4899" },
    Vadodara:    { color: "#10b981" }
};

// Gujarat district centroids for label placement (lat/lng)
const DISTRICT_CENTROIDS = {
    Ahmedabad:   [72.5714, 23.0225],
    Gandhinagar: [72.6369, 23.2156],
    Rajkot:      [70.8022, 22.3039],
    Surat:       [72.8311, 21.1702],
    Vadodara:    [73.1812, 22.3072]
};

// Name mapping from TopoJSON district names to our keys
// These are the names as they appear in the census/GADM data for Gujarat districts
const GUJARAT_DISTRICT_ID_MAP = {
    // GADM/census names → our internal names
    "Ahmadabad": "Ahmedabad",
    "Ahmedabad": "Ahmedabad",
    "Gandhi Nagar": "Gandhinagar",
    "Gandhinagar": "Gandhinagar",
    "Rājkot": "Rajkot",
    "Rajkot": "Rajkot",
    "Surat": "Surat",
    "Vadodara": "Vadodara",
    "Baroda": "Vadodara"
};
