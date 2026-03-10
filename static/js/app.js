/**
 * Main app initialization and state management.
 */

import { API } from './api.js';
import { ChatUI } from './chat.js';
import { ModelManager } from './models.js';
import { ResourceMonitor } from './resources.js';
import { SettingsUI } from './settings.js';

class AppState {
    constructor() {
        this.activeSessionId = null;
        this.sessions = [];
        this.registryModels = [];
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

class App {
    constructor() {
        this.api = new API();
        this.state = new AppState();
        this.chat = new ChatUI(this.api, this.state);
        this.models = new ModelManager(this.api, this.state);
        this.resources = new ResourceMonitor(this.api);
        this.settings = new SettingsUI(this.api, this.state);

        this._bindGlobalEvents();
        this._init();
    }

    async _init() {
        // Check health
        try {
            await this.api.getHealth();
        } catch {
            this.state.showToast('Cannot connect to server', 'error');
        }

        // Load sessions and model status
        await Promise.all([
            this._loadSessions(),
            this.models.refreshStatus(),
        ]);

        // Start resource monitoring
        this.resources.start();
    }

    _bindGlobalEvents() {
        // New chat
        document.getElementById('btn-new-chat').addEventListener('click', () => this._newChat());

        // Sidebar toggle (mobile)
        document.getElementById('sidebar-toggle').addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });

        // Close sidebar on overlay click (mobile)
        document.addEventListener('click', (e) => {
            const sidebar = document.getElementById('sidebar');
            const toggle = document.getElementById('sidebar-toggle');
            if (sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !toggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }

    async _loadSessions() {
        const listEl = document.getElementById('session-list');
        try {
            const sessions = await this.api.getSessions();
            this.state.sessions = sessions;
            listEl.innerHTML = '';

            if (!sessions.length) {
                listEl.innerHTML = '<div style="color:var(--text-secondary);font-size:12px;padding:4px 0;">No chats yet</div>';
                return;
            }

            for (const s of sessions) {
                const item = document.createElement('div');
                item.className = 'session-item' + (s.id === this.state.activeSessionId ? ' active' : '');
                item.dataset.id = s.id;

                const title = document.createElement('span');
                title.className = 'session-title';
                title.textContent = s.title;
                item.appendChild(title);

                const del = document.createElement('span');
                del.className = 'session-delete';
                del.textContent = '\u00D7';
                del.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this._deleteSession(s.id);
                });
                item.appendChild(del);

                // Click to switch
                item.addEventListener('click', () => this._switchSession(s.id, s.title));

                // Double-click to rename
                item.addEventListener('dblclick', (e) => {
                    e.stopPropagation();
                    this._startRename(item, s);
                });

                listEl.appendChild(item);
            }
        } catch (err) {
            this.state.showToast('Failed to load sessions: ' + err.message, 'error');
        }
    }

    async _newChat() {
        try {
            const session = await this.api.createSession('New Chat');
            this.state.showToast('Chat created', 'success');
            await this._loadSessions();
            await this._switchSession(session.id, session.title);
        } catch (err) {
            this.state.showToast('Failed to create chat: ' + err.message, 'error');
        }
    }

    async _switchSession(sessionId, title) {
        this.state.activeSessionId = sessionId;

        // Update UI highlight
        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', el.dataset.id === sessionId);
        });

        this.chat.setTitle(title);
        this.chat.setEnabled(true);

        // Load messages
        await this.chat.loadMessages(sessionId);

        // Connect WebSocket
        this.api.connectChat(sessionId, (data) => {
            this.chat.handleWsMessage(data);
        });
    }

    async _deleteSession(sessionId) {
        if (!confirm('Delete this chat?')) return;
        try {
            await this.api.deleteSession(sessionId);
            if (this.state.activeSessionId === sessionId) {
                this.state.activeSessionId = null;
                this.chat.clearMessages();
                this.chat.setTitle(null);
                this.chat.setEnabled(false);
                this.api.disconnect();
            }
            await this._loadSessions();
            this.state.showToast('Chat deleted', 'success');
        } catch (err) {
            this.state.showToast('Delete failed: ' + err.message, 'error');
        }
    }

    _startRename(itemEl, session) {
        const titleEl = itemEl.querySelector('.session-title');
        const originalTitle = titleEl.textContent;

        const input = document.createElement('input');
        input.className = 'session-rename-input';
        input.value = originalTitle;

        titleEl.replaceWith(input);
        input.focus();
        input.select();

        const finishRename = async () => {
            const newTitle = input.value.trim() || originalTitle;
            const span = document.createElement('span');
            span.className = 'session-title';
            span.textContent = newTitle;
            input.replaceWith(span);

            if (newTitle !== originalTitle) {
                try {
                    await this.api.updateSession(session.id, { title: newTitle });
                    session.title = newTitle;
                    if (this.state.activeSessionId === session.id) {
                        this.chat.setTitle(newTitle);
                    }
                } catch (err) {
                    span.textContent = originalTitle;
                    this.state.showToast('Rename failed: ' + err.message, 'error');
                }
            }
        };

        input.addEventListener('blur', finishRename);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                input.blur();
            } else if (e.key === 'Escape') {
                input.value = originalTitle;
                input.blur();
            }
        });
    }
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
