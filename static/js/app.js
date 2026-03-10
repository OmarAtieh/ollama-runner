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
        this.projects = [];
        this.activeProjectId = null;
        this.expandedProjects = new Set();
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

        // Load projects, sessions, and model status
        await Promise.all([
            this._loadProjects(),
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
            // Close context menu on click outside
            const ctxMenu = document.getElementById('session-context-menu');
            if (ctxMenu.style.display !== 'none' && !ctxMenu.contains(e.target)) {
                ctxMenu.style.display = 'none';
            }
        });

        // Project modal
        document.getElementById('btn-add-project').addEventListener('click', () => this._openProjectModal());
        const projModal = document.getElementById('project-modal');
        projModal.querySelectorAll('.modal-close, .modal-overlay').forEach(el => {
            el.addEventListener('click', () => { projModal.style.display = 'none'; });
        });
        document.getElementById('btn-create-project').addEventListener('click', () => this._createProject());

        // Color swatch selection
        document.getElementById('project-color-swatches').addEventListener('click', (e) => {
            const swatch = e.target.closest('.color-swatch');
            if (!swatch) return;
            document.querySelectorAll('#project-color-swatches .color-swatch').forEach(s => s.classList.remove('selected'));
            swatch.classList.add('selected');
        });
    }

    _createSessionItem(s) {
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

        // Right-click context menu to move to project
        item.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this._showSessionContextMenu(e, s);
        });

        return item;
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

            // Group sessions: those with project_id go under their project folder,
            // unsorted ones appear directly in the session list
            const projectSessions = {};
            const unsorted = [];
            for (const s of sessions) {
                if (s.project_id) {
                    if (!projectSessions[s.project_id]) projectSessions[s.project_id] = [];
                    projectSessions[s.project_id].push(s);
                } else {
                    unsorted.push(s);
                }
            }

            // Render project folders with their sessions
            for (const proj of this.state.projects) {
                const projSessions = projectSessions[proj.id] || [];
                if (projSessions.length === 0) continue;

                const folder = document.createElement('div');
                folder.className = 'project-folder';

                const header = document.createElement('div');
                header.className = 'project-header' + (this.state.activeProjectId === proj.id ? ' active' : '');
                const isExpanded = this.state.expandedProjects.has(proj.id);
                header.innerHTML = `
                    <span class="project-dot" style="background:${proj.color};"></span>
                    <span class="project-name">${this._escapeHtml(proj.name)}</span>
                    <span class="project-session-count">${projSessions.length}</span>
                    <span class="project-chevron ${isExpanded ? 'expanded' : ''}">&#9656;</span>
                `;
                header.addEventListener('click', () => {
                    const expanded = this.state.expandedProjects.has(proj.id);
                    if (expanded) {
                        this.state.expandedProjects.delete(proj.id);
                    } else {
                        this.state.expandedProjects.add(proj.id);
                    }
                    this.state.activeProjectId = expanded ? null : proj.id;
                    this._loadSessions();
                });
                folder.appendChild(header);

                if (isExpanded) {
                    const sessContainer = document.createElement('div');
                    sessContainer.className = 'project-sessions';
                    for (const s of projSessions) {
                        sessContainer.appendChild(this._createSessionItem(s));
                    }
                    folder.appendChild(sessContainer);
                }

                listEl.appendChild(folder);
            }

            // Render unsorted sessions
            for (const s of unsorted) {
                listEl.appendChild(this._createSessionItem(s));
            }
        } catch (err) {
            this.state.showToast('Failed to load sessions: ' + err.message, 'error');
        }
    }

    async _newChat() {
        try {
            const projectId = this.state.activeProjectId || null;
            const session = await this.api.createSession('New Chat', null, projectId);
            // If created in a project, make sure that project is expanded
            if (projectId) {
                this.state.expandedProjects.add(projectId);
            }
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

    // ── Projects ──────────────────────────────────────────────────

    async _loadProjects() {
        try {
            this.state.projects = await this.api.getProjects();
            this._renderProjectList();
        } catch (err) {
            // Projects may not be available on first run
            this.state.projects = [];
        }
    }

    _renderProjectList() {
        const listEl = document.getElementById('project-list');
        listEl.innerHTML = '';
        for (const proj of this.state.projects) {
            const item = document.createElement('div');
            item.className = 'project-header' + (this.state.activeProjectId === proj.id ? ' active' : '');
            item.innerHTML = `
                <span class="project-dot" style="background:${proj.color};"></span>
                <span class="project-name">${this._escapeHtml(proj.name)}</span>
                <span class="project-delete" title="Delete project">&times;</span>
            `;
            item.querySelector('.project-name').addEventListener('click', () => {
                // Toggle active project filter
                if (this.state.activeProjectId === proj.id) {
                    this.state.activeProjectId = null;
                    this.state.expandedProjects.delete(proj.id);
                } else {
                    this.state.activeProjectId = proj.id;
                    this.state.expandedProjects.add(proj.id);
                }
                this._loadSessions();
                this._renderProjectList();
            });
            item.querySelector('.project-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                this._deleteProject(proj.id);
            });
            listEl.appendChild(item);
        }
    }

    _openProjectModal() {
        const modal = document.getElementById('project-modal');
        document.getElementById('project-name-input').value = '';
        // Reset swatch selection to first
        document.querySelectorAll('#project-color-swatches .color-swatch').forEach((s, i) => {
            s.classList.toggle('selected', i === 0);
        });
        modal.style.display = 'flex';
        document.getElementById('project-name-input').focus();
    }

    async _createProject() {
        const name = document.getElementById('project-name-input').value.trim();
        if (!name) {
            this.state.showToast('Enter a project name', 'error');
            return;
        }
        const selectedSwatch = document.querySelector('#project-color-swatches .color-swatch.selected');
        const color = selectedSwatch ? selectedSwatch.dataset.color : '#6c8cff';
        try {
            await this.api.createProject({ name, color });
            document.getElementById('project-modal').style.display = 'none';
            this.state.showToast('Project created', 'success');
            await this._loadProjects();
            await this._loadSessions();
        } catch (err) {
            this.state.showToast('Failed to create project: ' + err.message, 'error');
        }
    }

    async _deleteProject(projectId) {
        if (!confirm('Delete this project? Sessions will be moved to Unsorted.')) return;
        try {
            await this.api.deleteProject(projectId);
            if (this.state.activeProjectId === projectId) {
                this.state.activeProjectId = null;
            }
            this.state.expandedProjects.delete(projectId);
            this.state.showToast('Project deleted', 'success');
            await this._loadProjects();
            await this._loadSessions();
        } catch (err) {
            this.state.showToast('Delete failed: ' + err.message, 'error');
        }
    }

    _showSessionContextMenu(event, session) {
        const menu = document.getElementById('session-context-menu');
        const itemsEl = document.getElementById('context-menu-projects');
        itemsEl.innerHTML = '';

        // "Unsorted" option
        const unsortedItem = document.createElement('div');
        unsortedItem.className = 'context-menu-item';
        unsortedItem.innerHTML = `<span class="project-dot" style="background:var(--text-muted);"></span> Unsorted`;
        unsortedItem.addEventListener('click', () => {
            this._moveSessionToProject(session.id, null);
            menu.style.display = 'none';
        });
        itemsEl.appendChild(unsortedItem);

        // Project options
        for (const proj of this.state.projects) {
            const item = document.createElement('div');
            item.className = 'context-menu-item';
            item.innerHTML = `<span class="project-dot" style="background:${proj.color};"></span> ${this._escapeHtml(proj.name)}`;
            item.addEventListener('click', () => {
                this._moveSessionToProject(session.id, proj.id);
                menu.style.display = 'none';
            });
            itemsEl.appendChild(item);
        }

        // Position menu
        menu.style.display = 'block';
        menu.style.left = event.clientX + 'px';
        menu.style.top = event.clientY + 'px';

        // Ensure menu stays within viewport
        requestAnimationFrame(() => {
            const rect = menu.getBoundingClientRect();
            if (rect.right > window.innerWidth) {
                menu.style.left = (window.innerWidth - rect.width - 8) + 'px';
            }
            if (rect.bottom > window.innerHeight) {
                menu.style.top = (window.innerHeight - rect.height - 8) + 'px';
            }
        });
    }

    async _moveSessionToProject(sessionId, projectId) {
        try {
            await this.api.updateSession(sessionId, { project_id: projectId });
            if (projectId) {
                this.state.expandedProjects.add(projectId);
            }
            await this._loadSessions();
        } catch (err) {
            this.state.showToast('Move failed: ' + err.message, 'error');
        }
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
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
