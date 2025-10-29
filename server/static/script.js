const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const API_URL = "https://9f36bb5f0f18.ngrok-free.app";

// Hàm thêm tin nhắn
function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  return msg;
}

// Gửi tin nhắn
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = userInput.value.trim();
  if (!prompt) return;

  addMessage(prompt, "user");
  userInput.value = "";

  const botMsg = addMessage("...", "bot");

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let text = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value);
      botMsg.textContent = text;
      chatBox.scrollTop = chatBox.scrollHeight;
    }

  } catch (err) {
    botMsg.textContent = "Lỗi kết nối server!";
  }
});
