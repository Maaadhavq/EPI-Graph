// EpiGraph-AI — SVG map path data only (no model data, that comes from API)
// Person B will add India + Gujarat SVG paths here

const INDIA_MAP = {
    viewBox: "0 0 800 900",
    gujaratPath: "",        // SVG path for Gujarat state outline
    otherStatesPath: "",    // SVG path for all other states (decorative)
    gujaratLabelPos: { x: 145, y: 480 }
};

const GUJARAT_MAP = {
    viewBox: "0 0 600 700",
    districts: {
        Ahmedabad:   { d: "", cx: 340, cy: 280 },
        Gandhinagar: { d: "", cx: 360, cy: 240 },
        Rajkot:      { d: "", cx: 150, cy: 320 },
        Surat:       { d: "", cx: 400, cy: 500 },
        Vadodara:    { d: "", cx: 420, cy: 380 }
    },
    otherDistrictsPath: ""  // Remaining districts as background
};
