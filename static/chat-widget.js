const convId = "conv-" + Math.floor(Math.random() * 999999);

function append(role, text, id=null) {
    const m = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = role;
    if (id) div.id = id;
    div.textContent = text;
    m.appendChild(div);
    m.scrollTop = m.scrollHeight;
    return div;
}

function sendMsg() {
    const txt = msgInput.value;
    if (!txt) return;
    append("user", txt);
    msgInput.value = "";
    msgInput.disabled = true;

    // Create loading bubble
    const loadingId = "loading-" + Date.now();
    append("bot loading", "Processing...", loadingId);

    fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: txt, conversation_id: convId})
    })
    .then(r => r.json())
    .then(j => {
        // Remove loading bubble
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.remove();

        if (j.needs_human) {
            append("bot", "AI a oprit. Un colegva prelua conversa■ia.");
        } else {
            append("bot", j.reply);
        }
    })
    .catch(err => {
        const loadingElem = document.getElementById(loadingId);
        if (loadingElem) loadingElem.remove();
        append("bot", "Error receiving response.");
        console.error(err);
    })
    .finally(() => {
        msgInput.disabled = false;
        msgInput.focus();
    });
}

window.onload = () => {
    fetch("/admin/settings", {headers:{Authorization:"Bearer "+localStorage.getItem("admin_token")}})
    .then(r => r.json())
    .then(j => append("bot", j.welcome || "Welcome!"));
};
