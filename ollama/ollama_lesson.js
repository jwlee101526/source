async function logJSONData() {
    const response = await fetch("http://localhost:11434/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "model": "gemma4:e4b",
            "messages": [
                {"role": "system", "content": "당신은 친절한 AI 어신스턴트입니다."},
                {"role": "user", "content" : "파이썬의 장점 3가지를 알려주세요"},
            ],
            stream: false
        })
    });
    const data = await response.json();
    console.log(
        data
    )
}

console.log("start ollama model serve")
logJSONData();