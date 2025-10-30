// === DOM ===
const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const messages = document.getElementById('messages');
const chatHistory = document.querySelector('.chat-history ul');
const imageInput = document.getElementById('image-upload');
const previewContainer = document.getElementById('image-preview'); // <<< THÊM
const previewImg = document.getElementById('preview-img');         // <<< THÊM
const removeBtn = document.getElementById('remove-image');         // <<< THÊM

const API_URL = "https://ca347c1cf367.ngrok-free.app";

// === ẢNH ===
let selectedImage = null;

imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    selectedImage = { file, dataUrl: ev.target.result };

    // HIỂN THỊ XEM TRƯỚC
    previewImg.src = ev.target.result;
    previewContainer.style.display = 'flex';

    updateSendButton();
  };
  reader.readAsDataURL(file);
});

// === XÓA ẢNH ===
removeBtn.addEventListener('click', () => {
  selectedImage = null;
  imageInput.value = '';
  previewContainer.style.display = 'none';
  updateSendButton();
});

function resetImage() {
  selectedImage = null;
  imageInput.value = '';
  previewContainer.style.display = 'none';
}

// === ENTER / SHIFT+ENTER ===
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event('submit'));
  }
});

// === CO GIÃN + NÚT GỬI ===
input.addEventListener('input', updateSendButton);

function updateSendButton() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';

  const canSend = input.value.trim() || selectedImage;
  sendBtn.disabled = !canSend;
  sendBtn.style.opacity = canSend ? '1' : '0.5';
}

// === GỬI ===
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const prompt = input.value.trim();
  const hasImage = !!selectedImage;
  if (!prompt && !hasImage) return;
  // === TẠO 1 DIV DUY NHẤT CHO USER ===
  const userMessage = document.createElement('div');
  userMessage.className = 'message user combined-message'; // 1 class

  let contentHTML = '';

  // Thêm ảnh (nếu có)
  if (hasImage) {
    contentHTML += `
      <img src="${selectedImage.dataUrl}" class="chat-user-image" alt="Ảnh bạn gửi" />
    `;
  }

  // Thêm text (nếu có)
  if (prompt) {
    contentHTML += `
      <div class="message-text">
        <p>${prompt.replace(/\n/g, '<br>')}</p>
      </div>
    `;
  }

  userMessage.innerHTML = `<div class="message-content">${contentHTML}</div>`;
  messages.appendChild(userMessage);

  // Cập nhật lịch sử
  addToHistory(prompt || '[Đã gửi ảnh]');

  const typing = showTyping();

  // === CHUẨN BỊ GỬI DỮ LIỆU ===
  const formData = new FormData();
  if (prompt) formData.append('prompt', prompt);
  if (hasImage) formData.append('image', selectedImage.file);

  
  input.value = '';
  input.style.height = 'auto';
  resetImage();
  updateSendButton();

  console.log("🟢 GỬI DỮ LIỆU:");
  for (let pair of formData.entries()) {
    console.log(pair[0] + ":", pair[1]);
  }

  try {
    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error(await res.text());

    typing.remove();
    const botMsg = addMessage('', 'ai');
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let text = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      botMsg.innerHTML = `<div class="message-content"><p>${text.replace(/\n/g, '<br>')}</p></div>`;
      messages.scrollTop = messages.scrollHeight;
    }
  } catch (err) {
    console.error("❌ LỖI:", err);
    typing.remove();
    addMessage(`<span style="color:red;">Lỗi: ${err.message}</span>`, 'ai');
  } finally {
    // === Reset sau khi gửi xong ===
    input.value = '';
    input.style.height = 'auto';
    resetImage();
    updateSendButton();
  }
});

// === HÀM ===
function addMessage(html, sender) {
  const div = document.createElement('div');
  div.className = `message ${sender}`;
  div.innerHTML = `<div class="message-content">${html}</div>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'message ai';
  div.innerHTML = `<div class="message-content"><div class="typing"><span></span><span></span><span></span></div></div>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function addToHistory(text) {
  const li = document.createElement('li');
  li.className = 'chat-item active';
  li.innerHTML = `<i class="fas fa-comment"></i><span>${text.substring(0, 35)}${text.length > 35 ? '...' : ''}</span>`;
  chatHistory.prepend(li);
}

// === XOÁ WELCOME ===
let first = true;
const oldAdd = addMessage;
addMessage = (html, sender) => {
  if (first && sender === 'user') {
    messages.innerHTML = '';
    first = false;
  }
  return oldAdd(html, sender);
};