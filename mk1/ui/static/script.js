document.addEventListener('DOMContentLoaded', () => {
    // Chat elements
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const chatWindow = document.getElementById('chat-window');
    const debugLog = document.getElementById('debug-log');

    // Template management elements
    const templateSelect = document.getElementById('template-select');
    const loadTemplateBtn = document.getElementById('load-template-btn');
    const deleteTemplateBtn = document.getElementById('delete-template-btn');
    const templateNameInput = document.getElementById('template-name-input');
    const saveTemplateBtn = document.getElementById('save-template-btn');
    const resetBtn = document.getElementById('reset-btn');

    // Initial load
    loadTemplates();

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        appendMessage('You', message, 'user-message');
        messageInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            appendMessage('Potato', data.response, 'bot-message');
            updateDebugLog(data.debug_log);

        } catch (error) {
            console.error('Error:', error);
            appendMessage('System', 'Sorry, I encountered an error. Please try again.', 'bot-message');
        }
    });

    function appendMessage(sender, message, messageClass) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', messageClass);
        
        const senderElement = document.createElement('strong');
        senderElement.textContent = sender;
        
        const messageTextElement = document.createElement('div');
        messageTextElement.textContent = message;

        messageElement.appendChild(senderElement);
        messageElement.appendChild(messageTextElement);
        
        chatMessages.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateDebugLog(logEntries) {
        debugLog.innerHTML = ''; // Clear previous log
        if (logEntries && Array.isArray(logEntries)) {
            logEntries.forEach(entry => {
                const p = document.createElement('p');
                p.textContent = entry;
                debugLog.appendChild(p);
            });
        }
    }

    // --- Template Management Functions ---

    async function loadTemplates() {
        try {
            const response = await fetch('/templates');
            const templates = await response.json();
            templateSelect.innerHTML = '';
            if (templates.length === 0) {
                const option = document.createElement('option');
                option.textContent = 'No templates saved';
                option.disabled = true;
                templateSelect.appendChild(option);
            } else {
                templates.forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    templateSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Failed to load templates:', error);
        }
    }

    saveTemplateBtn.addEventListener('click', async () => {
        const name = templateNameInput.value.trim();
        if (!name) {
            alert('Please enter a name for the template.');
            return;
        }
        try {
            const response = await fetch('/templates/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name }),
            });
            const result = await response.json();
            if (response.ok) {
                alert(result.success);
                templateNameInput.value = '';
                loadTemplates();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            alert(`Error saving template: ${error.message}`);
        }
    });

    loadTemplateBtn.addEventListener('click', () => handleLoadReset('load'));
    resetBtn.addEventListener('click', () => handleLoadReset('reset'));

    async function handleLoadReset(action) {
        const selectedTemplate = templateSelect.value;
        if (!selectedTemplate || templateSelect.disabled) {
            alert('No template selected.');
            return;
        }
        
        const confirmationText = action === 'reset' 
            ? `This will wipe all current progress and reset the bot's state to the '${selectedTemplate}' template. Are you sure?`
            : `This will load the '${selectedTemplate}' template. Current unsaved progress will be lost. Are you sure?`;

        if (!confirm(confirmationText)) {
            return;
        }

        try {
            const response = await fetch('/templates/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: selectedTemplate }),
            });
            const result = await response.json();
            if (response.ok) {
                alert(result.success);
                chatMessages.innerHTML = ''; // Clear chat window
                debugLog.innerHTML = ''; // Clear debug log
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            alert(`Error loading template: ${error.message}`);
        }
    }

    deleteTemplateBtn.addEventListener('click', async () => {
        const selectedTemplate = templateSelect.value;
        if (!selectedTemplate || templateSelect.disabled) {
            alert('No template selected.');
            return;
        }
        if (!confirm(`Are you sure you want to permanently delete the '${selectedTemplate}' template?`)) {
            return;
        }
        try {
             const response = await fetch('/templates/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: selectedTemplate }),
            });
            const result = await response.json();
            if (response.ok) {
                alert(result.success);
                loadTemplates();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            alert(`Error deleting template: ${error.message}`);
        }
    });
});
