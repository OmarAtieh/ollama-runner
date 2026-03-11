/**
 * Chat UI logic — message rendering, streaming, markdown.
 */

export class ChatUI {
    constructor(api, state) {
        this.api = api;
        this.state = state;

        this.messagesEl = document.getElementById('messages');
        this.inputEl = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('btn-send');
        this.typingEl = document.getElementById('typing-indicator');
        this.totalTokensEl = document.getElementById('total-tokens');
        this.chatTitleEl = document.getElementById('chat-title');

        this._currentAssistantEl = null;
        this._currentTokens = '';
        this._currentThinking = '';
        this._thinkingEl = null;
        this._streaming = false;
        this._userScrolledUp = false;

        this._initMarkdown();
        this._bindEvents();
    }

    _initMarkdown() {
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: (code, lang) => {
                    if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                        try { return hljs.highlight(code, { language: lang }).value; } catch {}
                    }
                    if (typeof hljs !== 'undefined') {
                        try { return hljs.highlightAuto(code).value; } catch {}
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
            });
        }
    }

    _bindEvents() {
        this.sendBtn.addEventListener('click', () => this._send());

        this.inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._send();
            }
        });

        // Auto-resize textarea
        this.inputEl.addEventListener('input', () => {
            this.inputEl.style.height = 'auto';
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 150) + 'px';
        });

        // Detect user scroll-up to stop auto-scroll
        this.messagesEl.addEventListener('scroll', () => {
            const { scrollTop, scrollHeight, clientHeight } = this.messagesEl;
            this._userScrolledUp = scrollHeight - scrollTop - clientHeight > 60;
        });
    }

    setEnabled(enabled) {
        this.inputEl.disabled = !enabled;
        this.sendBtn.disabled = !enabled;
    }

    setTitle(title) {
        this.chatTitleEl.textContent = title || 'Select or create a chat';
    }

    async loadMessages(sessionId) {
        this.messagesEl.innerHTML = '';
        this._currentAssistantEl = null;
        this._streaming = false;

        try {
            const messages = await this.api.getMessages(sessionId);
            for (const msg of messages) {
                this._appendMessage(msg.role, msg.content, msg.token_count, msg.tokens_per_second, msg.time_to_first_token_ms);
            }
            this._scrollToBottom(true);
            this._updateTokenCount(sessionId);
        } catch (err) {
            this.state.showToast('Failed to load messages: ' + err.message, 'error');
        }
    }

    clearMessages() {
        this.messagesEl.innerHTML = '';
        this.totalTokensEl.textContent = 'Total: 0 tokens';
    }

    async _updateTokenCount(sessionId) {
        try {
            const data = await this.api.getSessionTokens(sessionId);
            this.totalTokensEl.textContent = `Total: ${data.token_count || 0} tokens`;
        } catch {}
    }

    _appendMessage(role, content, tokenCount, tps, ttft) {
        const el = document.createElement('div');
        el.className = `message ${role}`;

        const roleLabel = document.createElement('div');
        roleLabel.className = 'message-role';
        roleLabel.textContent = role;
        el.appendChild(roleLabel);

        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';

        // Parse saved thinking blocks from stored messages
        if (role === 'assistant' && content && content.includes('<thinking>')) {
            const thinkMatch = content.match(/<thinking>\n?([\s\S]*?)\n?<\/thinking>\n*(.*)$/s);
            if (thinkMatch) {
                const thinkText = thinkMatch[1];
                const responseText = thinkMatch[2];
                const thinkBlock = document.createElement('details');
                thinkBlock.className = 'thinking-block';
                const summary = document.createElement('summary');
                summary.className = 'thinking-summary';
                const wordCount = thinkText.split(/\s+/).length;
                summary.textContent = `Thinking (${wordCount} words)`;
                thinkBlock.appendChild(summary);
                const thinkContent = document.createElement('div');
                thinkContent.className = 'thinking-content';
                thinkContent.innerHTML = this._renderMarkdown(thinkText);
                thinkBlock.appendChild(thinkContent);
                contentEl.appendChild(thinkBlock);
                if (responseText.trim()) {
                    const responseDiv = document.createElement('div');
                    responseDiv.className = 'response-content';
                    responseDiv.innerHTML = this._renderMarkdown(responseText);
                    contentEl.appendChild(responseDiv);
                }
            } else {
                contentEl.innerHTML = this._renderMarkdown(content);
            }
        } else {
            contentEl.innerHTML = this._renderMarkdown(content);
        }
        el.appendChild(contentEl);

        if (role === 'assistant' && tokenCount) {
            const statsEl = document.createElement('div');
            statsEl.className = 'message-stats';
            const parts = [`${tokenCount} tokens`];
            if (tps) parts.push(`${tps} t/s`);
            if (ttft) parts.push(`TTFT: ${ttft}ms`);
            statsEl.textContent = parts.join(' \u00B7 ');
            el.appendChild(statsEl);
        }

        this.messagesEl.appendChild(el);
        return el;
    }

    _renderMarkdown(text) {
        if (!text) return '';
        if (typeof marked !== 'undefined') {
            try { return marked.parse(text); } catch {}
        }
        // Fallback: escape HTML and preserve newlines
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }

    _scrollToBottom(force = false) {
        if (force || !this._userScrolledUp) {
            this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        }
    }

    _send() {
        const content = this.inputEl.value.trim();
        if (!content || this._streaming) return;
        if (!this.state.activeSessionId) {
            this.state.showToast('Create or select a chat first', 'error');
            return;
        }

        // Add user message to UI
        this._appendMessage('user', content);
        this._scrollToBottom(true);

        // Clear input
        this.inputEl.value = '';
        this.inputEl.style.height = 'auto';

        // Send via WebSocket
        this.api.sendMessage(content);
        this._streaming = true;
        this._showTyping(true);
        this.sendBtn.disabled = true;
    }

    // Called by app.js when WS message arrives
    handleWsMessage(data) {
        switch (data.type) {
            case 'start':
                this._currentTokens = '';
                this._currentThinking = '';
                this._thinkingEl = null;
                this._currentAssistantEl = this._appendMessage('assistant', '');
                this._showTyping(false);
                break;

            case 'thinking_start': {
                // Create a collapsible thinking block inside the message
                if (this._currentAssistantEl) {
                    const contentEl = this._currentAssistantEl.querySelector('.message-content');
                    if (contentEl) {
                        const thinkBlock = document.createElement('details');
                        thinkBlock.className = 'thinking-block';
                        thinkBlock.open = true;
                        const summary = document.createElement('summary');
                        summary.className = 'thinking-summary';
                        summary.textContent = 'Thinking...';
                        thinkBlock.appendChild(summary);
                        const thinkContent = document.createElement('div');
                        thinkContent.className = 'thinking-content';
                        thinkBlock.appendChild(thinkContent);
                        contentEl.appendChild(thinkBlock);
                        this._thinkingEl = thinkContent;
                    }
                }
                break;
            }

            case 'thinking':
                if (data.content && this._thinkingEl) {
                    this._currentThinking += data.content;
                    this._thinkingEl.innerHTML = this._renderMarkdown(this._currentThinking);
                    this._scrollToBottom();
                }
                break;

            case 'thinking_end':
                if (this._thinkingEl) {
                    // Final render of thinking
                    this._thinkingEl.innerHTML = this._renderMarkdown(this._currentThinking);
                    // Update summary with token count
                    const details = this._thinkingEl.closest('.thinking-block');
                    if (details) {
                        const summary = details.querySelector('.thinking-summary');
                        const wordCount = this._currentThinking.split(/\s+/).length;
                        summary.textContent = `Thinking (${wordCount} words)`;
                        details.open = false; // Collapse thinking when done
                    }
                }
                this._thinkingEl = null;
                break;

            case 'token':
                if (data.content) {
                    this._currentTokens += data.content;
                    const contentEl = this._currentAssistantEl?.querySelector('.message-content');
                    if (contentEl) {
                        // Preserve thinking block, update response after it
                        const thinkBlock = contentEl.querySelector('.thinking-block');
                        if (thinkBlock) {
                            // Find or create a response div after the thinking block
                            let responseDiv = contentEl.querySelector('.response-content');
                            if (!responseDiv) {
                                responseDiv = document.createElement('div');
                                responseDiv.className = 'response-content';
                                contentEl.appendChild(responseDiv);
                            }
                            responseDiv.innerHTML = this._renderMarkdown(this._currentTokens);
                        } else {
                            contentEl.innerHTML = this._renderMarkdown(this._currentTokens);
                        }
                    }
                    this._scrollToBottom();
                }
                break;

            case 'done':
                if (this._currentAssistantEl && data.stats) {
                    // Re-render final content
                    const contentEl = this._currentAssistantEl.querySelector('.message-content');
                    if (contentEl) {
                        const thinkBlock = contentEl.querySelector('.thinking-block');
                        if (thinkBlock) {
                            // Keep thinking block, update response
                            let responseDiv = contentEl.querySelector('.response-content');
                            if (!responseDiv) {
                                responseDiv = document.createElement('div');
                                responseDiv.className = 'response-content';
                                contentEl.appendChild(responseDiv);
                            }
                            responseDiv.innerHTML = this._renderMarkdown(data.content || this._currentTokens);
                        } else {
                            contentEl.innerHTML = this._renderMarkdown(data.content || this._currentTokens);
                        }
                    }

                    // Add stats
                    const statsEl = document.createElement('div');
                    statsEl.className = 'message-stats';
                    const s = data.stats;
                    const parts = [`${s.token_count} tokens`];
                    if (s.tokens_per_second) parts.push(`${s.tokens_per_second} t/s`);
                    if (s.time_to_first_token_ms) parts.push(`TTFT: ${s.time_to_first_token_ms}ms`);
                    statsEl.textContent = parts.join(' \u00B7 ');
                    this._currentAssistantEl.appendChild(statsEl);

                    // Highlight code blocks
                    if (typeof hljs !== 'undefined') {
                        this._currentAssistantEl.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    }
                }

                this._streaming = false;
                this._currentAssistantEl = null;
                this._currentTokens = '';
                this._currentThinking = '';
                this._thinkingEl = null;
                this.sendBtn.disabled = false;
                this._scrollToBottom(true);

                // Update token count
                if (this.state.activeSessionId) {
                    this._updateTokenCount(this.state.activeSessionId);
                }
                break;

            case 'error':
                this._showTyping(false);
                this._streaming = false;
                this.sendBtn.disabled = false;
                this.state.showToast(data.content || 'Chat error', 'error');
                break;
        }
    }

    _showTyping(show) {
        this.typingEl.style.display = show ? 'flex' : 'none';
        if (show) this._scrollToBottom(true);
    }
}
