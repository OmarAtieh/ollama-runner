/**
 * Model picker modal and management logic.
 */

export class ModelManager {
    constructor(api, state) {
        this.api = api;
        this.state = state;

        this.modal = document.getElementById('model-modal');
        this.scannedListEl = document.getElementById('scanned-models-list');
        this.registryListEl = document.getElementById('registry-list');
        this.configPanel = document.getElementById('model-config-panel');
        this.binarySection = document.getElementById('binary-section');

        this._selectedRegistryId = null;
        this._estimateTimeout = null;

        this._bindEvents();
    }

    _bindEvents() {
        // Open modal
        document.getElementById('btn-model-picker').addEventListener('click', () => this.open());

        // Close modal
        this.modal.querySelectorAll('.modal-close, .modal-overlay').forEach(el => {
            el.addEventListener('click', () => this.close());
        });

        // Scan button
        document.getElementById('btn-scan-models').addEventListener('click', () => this._scanModels());

        // Download binary
        document.getElementById('btn-download-binary').addEventListener('click', () => this._downloadBinary());

        // Sidebar load/unload
        document.getElementById('btn-load-model').addEventListener('click', () => this._loadFromSidebar());
        document.getElementById('btn-unload-model').addEventListener('click', () => this._unloadModel());

        // Config panel sliders
        const sliders = ['slider-gpu-layers', 'slider-ctx-length', 'slider-temp', 'slider-topp'];
        const valueEls = ['gpu-layers-value', 'ctx-length-value', 'temp-value', 'topp-value'];
        sliders.forEach((id, i) => {
            document.getElementById(id).addEventListener('input', (e) => {
                document.getElementById(valueEls[i]).textContent = e.target.value;
                this._scheduleEstimate();
            });
        });

        // Save config
        document.getElementById('btn-save-model-config').addEventListener('click', () => this._saveModelConfig());

        // Load from config panel
        document.getElementById('btn-load-from-config').addEventListener('click', () => this._loadFromConfig());

        // Delete model
        document.getElementById('btn-delete-model').addEventListener('click', () => this._deleteSelectedModel());
    }

    async open() {
        this.modal.style.display = 'flex';
        await Promise.all([
            this._checkBinary(),
            this._loadRegistry(),
        ]);
    }

    close() {
        this.modal.style.display = 'none';
    }

    async _checkBinary() {
        try {
            const status = await this.api.getBinaryStatus();
            if (!status.available) {
                this.binarySection.style.display = 'block';
                document.getElementById('binary-status-text').textContent =
                    'llama-server not found. Download required.';
            } else {
                this.binarySection.style.display = 'none';
            }
        } catch (err) {
            this.binarySection.style.display = 'block';
            document.getElementById('binary-status-text').textContent =
                'Could not check binary status: ' + err.message;
        }
    }

    async _downloadBinary() {
        const btn = document.getElementById('btn-download-binary');
        const prog = document.getElementById('binary-progress');
        const bar = document.getElementById('binary-progress-bar');
        btn.disabled = true;
        prog.style.display = 'block';
        bar.style.width = '50%';

        try {
            await this.api.downloadBinary();
            bar.style.width = '100%';
            this.state.showToast('llama-server downloaded successfully', 'success');
            this.binarySection.style.display = 'none';
        } catch (err) {
            this.state.showToast('Download failed: ' + err.message, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    async _scanModels() {
        this.scannedListEl.innerHTML = '<div style="color:var(--text-secondary);font-size:12px;">Scanning...</div>';
        try {
            const models = await this.api.scanModels();
            if (!models.length) {
                this.scannedListEl.innerHTML = '<div style="color:var(--text-secondary);font-size:12px;">No GGUF files found.</div>';
                return;
            }
            this.scannedListEl.innerHTML = '';
            for (const m of models) {
                const item = document.createElement('div');
                item.className = 'model-list-item';
                const sizeGB = m.file_size_mb ? (m.file_size_mb / 1024).toFixed(1) + ' GB' : '';
                item.innerHTML = `
                    <div class="model-list-item-info">
                        <div class="model-list-item-name">${this._escapeHtml(m.model_name)}</div>
                        <div class="model-list-item-meta">${sizeGB} ${m.quantization || ''} ctx:${m.context_length || '?'}</div>
                    </div>
                    <button class="btn btn-sm btn-accent">Add</button>
                `;
                item.querySelector('button').addEventListener('click', (e) => {
                    e.stopPropagation();
                    this._addToRegistry(m);
                });
                this.scannedListEl.appendChild(item);
            }
        } catch (err) {
            this.scannedListEl.innerHTML = '';
            this.state.showToast('Scan failed: ' + err.message, 'error');
        }
    }

    async _addToRegistry(scanned) {
        try {
            await this.api.addModel({
                name: scanned.model_name,
                path: scanned.file_path,
                context_default: scanned.context_length || 4096,
                context_max: scanned.context_length || 32768,
            });
            this.state.showToast('Model added to registry', 'success');
            await this._loadRegistry();
        } catch (err) {
            this.state.showToast('Failed to add: ' + err.message, 'error');
        }
    }

    async _loadRegistry() {
        try {
            const models = await this.api.getRegistry();
            this.state.registryModels = models;
            this.registryListEl.innerHTML = '';

            if (!models.length) {
                this.registryListEl.innerHTML = '<div style="color:var(--text-secondary);font-size:12px;">No models registered. Scan and add models.</div>';
                this.configPanel.style.display = 'none';
                return;
            }

            for (const m of models) {
                const item = document.createElement('div');
                item.className = 'model-list-item' + (m.id === this._selectedRegistryId ? ' selected' : '');
                item.innerHTML = `
                    <div class="model-list-item-info">
                        <div class="model-list-item-name">${this._escapeHtml(m.name)}</div>
                        <div class="model-list-item-meta">GPU layers: ${m.gpu_layers} | ctx: ${m.context_default}</div>
                    </div>
                `;
                item.addEventListener('click', () => this._selectRegistryModel(m));
                this.registryListEl.appendChild(item);
            }
        } catch (err) {
            this.state.showToast('Failed to load registry: ' + err.message, 'error');
        }
    }

    _selectRegistryModel(model) {
        this._selectedRegistryId = model.id;
        this.configPanel.style.display = 'block';
        document.getElementById('config-model-name').textContent = model.name;

        document.getElementById('slider-gpu-layers').value = model.gpu_layers;
        document.getElementById('gpu-layers-value').textContent = model.gpu_layers;

        document.getElementById('slider-ctx-length').value = model.context_default;
        document.getElementById('ctx-length-value').textContent = model.context_default;

        document.getElementById('slider-temp').value = model.temperature;
        document.getElementById('temp-value').textContent = model.temperature;

        document.getElementById('slider-topp').value = model.top_p;
        document.getElementById('topp-value').textContent = model.top_p;

        // Highlight selected
        this.registryListEl.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('selected'));
        // Re-find and highlight
        const items = this.registryListEl.querySelectorAll('.model-list-item');
        const idx = this.state.registryModels.findIndex(m => m.id === model.id);
        if (idx >= 0 && items[idx]) items[idx].classList.add('selected');

        this._fetchEstimate();
    }

    _scheduleEstimate() {
        clearTimeout(this._estimateTimeout);
        this._estimateTimeout = setTimeout(() => this._fetchEstimate(), 300);
    }

    async _fetchEstimate() {
        if (!this._selectedRegistryId) return;
        const gpuLayers = parseInt(document.getElementById('slider-gpu-layers').value);
        const ctxLength = parseInt(document.getElementById('slider-ctx-length').value);
        const estimateEl = document.getElementById('resource-estimate');

        try {
            const est = await this.api.estimateResources(this._selectedRegistryId, gpuLayers, ctxLength);
            let html = '';
            if (est.vram_needed_mb != null) html += `VRAM: ~${est.vram_needed_mb} MB\n`;
            if (est.ram_needed_mb != null) html += `RAM: ~${est.ram_needed_mb} MB\n`;
            if (est.feasible != null) {
                html += est.feasible ? 'Status: Feasible' : 'Status: May not fit!';
            }
            if (est.warning) html += `\nWarning: ${est.warning}`;
            estimateEl.textContent = html || 'No estimate available';
            estimateEl.style.display = 'block';
        } catch {
            estimateEl.textContent = 'Estimate unavailable';
            estimateEl.style.display = 'block';
        }
    }

    async _saveModelConfig() {
        if (!this._selectedRegistryId) return;
        try {
            await this.api.updateModel(this._selectedRegistryId, {
                gpu_layers: parseInt(document.getElementById('slider-gpu-layers').value),
                context_default: parseInt(document.getElementById('slider-ctx-length').value),
                temperature: parseFloat(document.getElementById('slider-temp').value),
                top_p: parseFloat(document.getElementById('slider-topp').value),
            });
            this.state.showToast('Model config saved', 'success');
            await this._loadRegistry();
        } catch (err) {
            this.state.showToast('Save failed: ' + err.message, 'error');
        }
    }

    async _loadFromConfig() {
        if (!this._selectedRegistryId) return;
        const ctxLength = parseInt(document.getElementById('slider-ctx-length').value);
        this.close();
        await this._loadModel(this._selectedRegistryId, ctxLength);
    }

    async _loadFromSidebar() {
        // Load whatever model is selected or first in registry
        if (this._selectedRegistryId) {
            await this._loadModel(this._selectedRegistryId);
        } else if (this.state.registryModels?.length) {
            await this._loadModel(this.state.registryModels[0].id);
        } else {
            this.state.showToast('No model selected. Open Model Manager first.', 'error');
        }
    }

    async _loadModel(modelId, contextLength) {
        const progressContainer = document.getElementById('model-progress-container');
        const progressBar = document.getElementById('model-progress-bar');
        const progressText = document.getElementById('model-progress-text');
        const loadBtn = document.getElementById('btn-load-model');
        const unloadBtn = document.getElementById('btn-unload-model');

        loadBtn.disabled = true;
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting...';

        try {
            await this.api.loadModel(modelId, contextLength);

            // Poll status
            let loaded = false;
            let polls = 0;
            while (!loaded && polls < 120) {
                await this._sleep(500);
                polls++;
                const status = await this.api.getModelStatus();

                if (status.loading) {
                    const pct = Math.round((status.load_progress || 0) * 100);
                    progressBar.style.width = pct + '%';
                    progressText.textContent = `Loading... ${pct}%`;
                } else if (status.loaded) {
                    loaded = true;
                    progressBar.style.width = '100%';
                    progressText.textContent = 'Loaded';
                    this._updateModelDisplay(status);
                    this.state.showToast('Model loaded', 'success');
                } else if (status.error) {
                    throw new Error(status.error);
                } else {
                    // Might be loaded already
                    loaded = true;
                    this._updateModelDisplay(status);
                }
            }
        } catch (err) {
            this.state.showToast('Load failed: ' + err.message, 'error');
            progressText.textContent = 'Failed';
        } finally {
            loadBtn.disabled = false;
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 2000);
        }
    }

    async _unloadModel() {
        try {
            await this.api.unloadModel();
            document.getElementById('current-model-name').textContent = 'No model loaded';
            document.getElementById('btn-unload-model').disabled = true;
            this.state.showToast('Model unloaded', 'success');
        } catch (err) {
            this.state.showToast('Unload failed: ' + err.message, 'error');
        }
    }

    async _deleteSelectedModel() {
        if (!this._selectedRegistryId) return;
        if (!confirm('Remove this model from the registry?')) return;
        try {
            await this.api.deleteModel(this._selectedRegistryId);
            this._selectedRegistryId = null;
            this.configPanel.style.display = 'none';
            this.state.showToast('Model removed', 'success');
            await this._loadRegistry();
        } catch (err) {
            this.state.showToast('Delete failed: ' + err.message, 'error');
        }
    }

    _updateModelDisplay(status) {
        const nameEl = document.getElementById('current-model-name');
        const unloadBtn = document.getElementById('btn-unload-model');
        const loadBtn = document.getElementById('btn-load-model');

        if (status.loaded) {
            nameEl.textContent = status.current_model?.name || 'Model loaded';
            unloadBtn.disabled = false;
            loadBtn.disabled = false;
        } else {
            nameEl.textContent = 'No model loaded';
            unloadBtn.disabled = true;
            loadBtn.disabled = false;
        }
    }

    async refreshStatus() {
        try {
            const status = await this.api.getModelStatus();
            this._updateModelDisplay(status);

            // Also load registry in background
            const models = await this.api.getRegistry();
            this.state.registryModels = models;
            document.getElementById('btn-load-model').disabled = models.length === 0;
        } catch {}
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
