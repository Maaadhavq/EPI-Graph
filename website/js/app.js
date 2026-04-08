// EpiGraph-AI — Shared utilities, navigation, API fetch helper
// Person B will build this out

async function apiFetch(endpoint) {
    const res = await fetch(endpoint);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}
