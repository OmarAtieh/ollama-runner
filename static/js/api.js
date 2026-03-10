/**
 * API client for OllamaRunner backend — REST + WebSocket.
 */

export class API {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.ws = null;
        this._reconnectAttempts = 0;
        this._maxReconnectAttempts = 5;
        this._reconnectDelay = 1000;
    }

    // ── REST helpers ──────────────────────────────────────────────

    async _fetch(path, options = {}) {
        const url = `${this.baseUrl}${path}`;
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };
        const opts = { ...defaults, ...options };
        const resp = await fetch(url, opts);
        if (!resp.ok) {
            let detail = '';
            try {
                const body = await resp.json();
                detail = body.detail || JSON.stringify(body);
            } catch {
                detail = await resp.text();
            }
            throw new Error(`${resp.status}: ${detail}`);
        }
        if (resp.status === 204) return null;
        return resp.json();
    }

    async _get(path) {
        return this._fetch(path);
    }

    async _post(path, data) {
        return this._fetch(path, {
            method: 'POST',
            body: data !== undefined ? JSON.stringify(data) : undefined,
        });
    }

    async _put(path, data) {
        return this._fetch(path, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    }

    async _delete(path) {
        return this._fetch(path, { method: 'DELETE' });
    }

    // ── System ────────────────────────────────────────────────────

    async getHealth() {
        return this._get('/api/health');
    }

    async getResources() {
        return this._get('/api/system/resources');
    }

    async getBinaryStatus() {
        return this._get('/api/system/binary/status');
    }

    async downloadBinary() {
        return this._post('/api/system/binary/download');
    }

    // ── Models ────────────────────────────────────────────────────

    async scanModels() {
        return this._get('/api/models/scan');
    }

    async getRegistry() {
        return this._get('/api/models/registry');
    }

    async addModel(data) {
        return this._post('/api/models/registry', data);
    }

    async updateModel(id, data) {
        return this._put(`/api/models/registry/${id}`, data);
    }

    async deleteModel(id) {
        return this._delete(`/api/models/registry/${id}`);
    }

    async loadModel(id, contextLength) {
        const body = {};
        if (contextLength != null) body.context_length = contextLength;
        return this._post(`/api/models/load/${id}`, body);
    }

    async unloadModel() {
        return this._post('/api/models/unload');
    }

    async getModelStatus() {
        return this._get('/api/models/status');
    }

    async estimateResources(id, gpuLayers, contextLength) {
        const params = new URLSearchParams();
        if (gpuLayers != null) params.set('gpu_layers', gpuLayers);
        if (contextLength != null) params.set('context_length', contextLength);
        return this._get(`/api/models/estimate/${id}?${params}`);
    }

    async recommendLayers(modelPath) {
        return this._get(`/api/models/recommend-layers?model_path=${encodeURIComponent(modelPath)}`);
    }

    // ── Sessions ──────────────────────────────────────────────────

    async getSessions() {
        return this._get('/api/sessions/');
    }

    async createSession(title, modelId, projectId) {
        const body = { title, model_id: modelId || null };
        if (projectId) body.project_id = projectId;
        return this._post('/api/sessions/', body);
    }

    async deleteSession(id) {
        return this._delete(`/api/sessions/${id}`);
    }

    async updateSession(id, data) {
        return this._put(`/api/sessions/${id}`, data);
    }

    async getMessages(sessionId) {
        return this._get(`/api/sessions/${sessionId}/messages`);
    }

    async getSessionTokens(sessionId) {
        return this._get(`/api/sessions/${sessionId}/tokens`);
    }

    // ── Projects ──────────────────────────────────────────────────

    async getProjects() {
        return this._get('/api/projects/');
    }

    async createProject(data) {
        return this._post('/api/projects/', data);
    }

    async updateProject(id, data) {
        return this._put(`/api/projects/${id}`, data);
    }

    async deleteProject(id) {
        return this._delete(`/api/projects/${id}`);
    }

    // ── Config ────────────────────────────────────────────────────

    async getConfig() {
        return this._get('/api/config/');
    }

    async updateConfig(data) {
        return this._put('/api/config/', data);
    }

    async getPromptFile(filename) {
        return this._get(`/api/config/prompt/${filename}`);
    }

    async savePromptFile(filename, content) {
        return this._put(`/api/config/prompt/${filename}`, { content });
    }

    // ── WebSocket ─────────────────────────────────────────────────

    connectChat(sessionId, onMessage) {
        this.disconnect();
        this._reconnectAttempts = 0;

        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/chat/${sessionId}`;

        this._wsSessionId = sessionId;
        this._wsOnMessage = onMessage;

        this.ws = new WebSocket(url);

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        this.ws.onopen = () => {
            this._reconnectAttempts = 0;
        };

        this.ws.onclose = () => {
            this._tryReconnect();
        };

        this.ws.onerror = (err) => {
            console.error('WS error:', err);
        };

        return this.ws;
    }

    sendMessage(content) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ content }));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.onclose = null; // prevent reconnect
            this.ws.close();
            this.ws = null;
        }
    }

    _tryReconnect() {
        if (this._reconnectAttempts >= this._maxReconnectAttempts) return;
        if (!this._wsSessionId || !this._wsOnMessage) return;

        this._reconnectAttempts++;
        const delay = this._reconnectDelay * Math.pow(2, this._reconnectAttempts - 1);

        setTimeout(() => {
            if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                this.connectChat(this._wsSessionId, this._wsOnMessage);
            }
        }, delay);
    }
}
