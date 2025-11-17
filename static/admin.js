const API = "/admin/settings";

function loadSettings() {
    const token = localStorage.getItem("admin_token");
    if (!token) return window.location = "/login";

    fetch(API, {headers: {Authorization: "Bearer " + token}})
        .then(r => r.json())
        .then(j => {
            welcome.value = j.welcome;
            fallback.value = j.fallback;
            tone.value = j.tone;
        });
}

function saveSettings() {
    const token = localStorage.getItem("admin_token");

    fetch(API, {
        method: "POST",
        headers: {
            Authorization: "Bearer " + token,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            welcome: welcome.value,
            fallback: fallback.value,
            tone: tone.value
        })
    })
    .then(r => r.json())
    .then(j => {
        status.textContent = "Saved!";
    });
}

function logout() {
    localStorage.removeItem("admin_token");
    window.location = "/login";
}
