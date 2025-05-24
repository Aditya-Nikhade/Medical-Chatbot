document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');

    // Add timestamp to messages
    function getTimestamp() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    // Add loading indicator
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot');
        loadingDiv.innerHTML = `
            <div class="message-content">
                <i class="fas fa-robot"></i>
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loadingDiv;
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        const icon = sender === 'user' ? 'fa-user' : 'fa-robot';
        const timestamp = getTimestamp();
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <i class="fas ${icon}"></i>
                <div class="text">${text}</div>
            </div>
            <div class="timestamp">${timestamp}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user');
        userInput.value = '';

        // Add loading indicator
        const loadingIndicator = addLoadingIndicator();

        try {
            // Send message to server
            const response = await fetch('/get', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `msg=${encodeURIComponent(message)}`
            });

            // Remove loading indicator
            loadingIndicator.remove();

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const botResponse = await response.text();
            addMessage(botResponse, 'bot');
        } catch (error) {
            // Remove loading indicator
            loadingIndicator.remove();
            
            console.error('Error:', error);
            addMessage(
                'I apologize, but I encountered an error while processing your request. Please try again.',
                'bot'
            );
        }
    });

    // Handle Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Focus input on load
    userInput.focus();
}); 