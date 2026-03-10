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
        document.getElementById('btn-model-picker').addEventListener('click', (e) => {
            e.preventDefault();
            this.open();
        });

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

        // Sidebar model section collapse/expand
        document.getElementById('model-section-toggle').addEventListener('click', () => this._toggleModelExpanded());

        // Sidebar inline sliders
        this._bindSidebarSlider('sidebar-slider-temp', 'sidebar-temp-val', false);
        this._bindSidebarSlider('sidebar-slider-ctx', 'sidebar-ctx-val', true);
        this._bindSidebarSlider('sidebar-slider-gpu', 'sidebar-gpu-val', true);

        // Sidebar reload button
        document.getElementById('btn-sidebar-reload').addEventListener('click', () => this._sidebarReload());

        // Slider + input sync for model config
        this._syncSliderInput('slider-gpu-layers', 'input-gpu-layers', 'gpu-layers-value', true);
        this._syncSliderInput('slider-ctx-length', 'input-ctx-length', 'ctx-length-value', true);
        this._syncSliderInput('slider-temp', 'input-temp', 'temp-value', false);
        this._syncSliderInput('slider-topp', 'input-topp', 'topp-value', false);

        // Save config
        document.getElementById('btn-save-model-config').addEventListener('click', () => this._saveModelConfig());

        // Load from config panel
        document.getElementById('btn-load-from-config').addEventListener('click', () => this._loadFromConfig());

        // Delete model
        document.getElementById('btn-delete-model').addEventListener('click', () => this._deleteSelectedModel());
    }

    _toggleModelExpanded() {
        const expanded = document.getElementById('sidebar-model-expanded');
        const chevron = document.querySelector('.model-toggle-chevron');
        const isVisible = expanded.style.display !== 'none';
        expanded.style.display = isVisible ? 'none' : 'flex';
        chevron.classList.toggle('expanded', !isVisible);
    }

    _bindSidebarSlider(sliderId, valId, isInt) {
        const slider = document.getElementById(sliderId);
        const valEl = document.getElementById(valId);
        slider.addEventListener('input', () => {
            const v = isInt ? parseInt(slider.value) : parseFloat(slider.value);
            valEl.textContent = v;
        });
    }

    async _sidebarReload() {
        if (!this._selectedRegistryId) {
            this.state.showToast('No model selected', 'error');
            return;
        }
        // Save inline slider values to model config, then reload
        try {
            await this.api.updateModel(this._selectedRegistryId, {
                temperature: parseFloat(document.getElementById('sidebar-slider-temp').value),
                context_default: parseInt(document.getElementById('sidebar-slider-ctx').value),
                gpu_layers: parseInt(document.getElementById('sidebar-slider-gpu').value),
            });
            this.state.showToast('Config saved, reloading...', 'success');
            const ctxLength = parseInt(document.getElementById('sidebar-slider-ctx').value);
            await this._loadModel(this._selectedRegistryId, ctxLength);
        } catch (err) {
            this.state.showToast('Reload failed: ' + err.message, 'error');
        }
    }

    /**
     * Sync a range slider with a number input and a display span.
     */
    _syncSliderInput(sliderId, inputId, valueId, isInt) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        const valueEl = document.getElementById(valueId);

        const format = (v) => isInt ? parseInt(v) : parseFloat(v);

        slider.addEventListener('input', () => {
            const v = format(slider.value);
            input.value = v;
            valueEl.textContent = v;
            this._scheduleEstimate();
        });

        input.addEventListener('input', () => {
            let v = format(input.value);
            const min = format(slider.min);
            const max = format(slider.max);
            if (v < min) v = min;
            if (v > max) v = max;
            slider.value = v;
            valueEl.textContent = v;
            this._scheduleEstimate();
        });

        input.addEventListener('blur', () => {
            // Clamp on blur
            let v = format(input.value);
            const min = format(slider.min);
            const max = format(slider.max);
            if (isNaN(v)) v = format(slider.value);
            if (v < min) v = min;
            if (v > max) v = max;
            input.value = v;
            slider.value = v;
            valueEl.textContent = v;
        });
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
        const statusText = document.getElementById('binary-status-text');
        btn.disabled = true;
        prog.style.display = 'block';
        bar.style.width = '0%';

        try {
            await this.api.downloadBinary();
            let done = false;
            while (!done) {
                await this._sleep(1000);
                const status = await this.api.getBinaryStatus();
                const pct = Math.round((status.progress || 0) * 100);
                bar.style.width = pct + '%';
                statusText.textContent = `${status.status}: ${pct}%`;
                if (status.status === 'ready' || status.available) {
                    done = true;
                    bar.style.width = '100%';
                    statusText.textContent = 'Download complete';
                    this.state.showToast('llama-server downloaded successfully', 'success');
                    this.binarySection.style.display = 'none';
                } else if (status.status === 'error') {
                    done = true;
                    this.state.showToast('Download failed', 'error');
                    statusText.textContent = 'Download failed';
                }
            }
        } catch (err) {
            this.state.showToast('Download failed: ' + err.message, 'error');
        } finally {
            btn.disabled = false;
        }
    }

    async _scanModels() {
        this.scannedListEl.innerHTML = '<div class="empty-state">Scanning...</div>';
        try {
            const models = await this.api.scanModels();
            if (!models.length) {
                this.scannedListEl.innerHTML = '<div class="empty-state">No GGUF files found.</div>';
                return;
            }

            // Build set of already-registered file paths for quick lookup
            const registeredPaths = new Set(
                (this.state.registryModels || []).map(r => r.path)
            );

            this.scannedListEl.innerHTML = '';
            for (const m of models) {
                const alreadyAdded = registeredPaths.has(m.file_path);
                const item = document.createElement('div');
                item.className = 'model-list-item' + (alreadyAdded ? ' added' : '');
                const sizeGB = m.file_size_mb ? (m.file_size_mb / 1024).toFixed(1) + ' GB' : '';
                const displayName = m.display_name || m.model_name || 'Unknown';
                const params = m.parameter_count ? this._formatParams(m.parameter_count) : '';
                const arch = m.architecture || '';

                let metaHtml = '';
                if (sizeGB) metaHtml += `<span class="meta-tag">${sizeGB}</span>`;
                if (m.quantization && m.quantization !== 'unknown') metaHtml += `<span class="meta-tag">${this._escapeHtml(m.quantization)}</span>`;
                if (params) metaHtml += `<span class="meta-tag">${params}</span>`;
                if (arch) metaHtml += `<span class="meta-tag">${this._escapeHtml(arch)}</span>`;
                if (m.context_length) metaHtml += `<span class="meta-tag">ctx ${m.context_length.toLocaleString()}</span>`;

                // Capability badges
                let capsHtml = '';
                if (m.capabilities && m.capabilities.length) {
                    capsHtml = `<div class="model-list-item-caps">${m.capabilities.map(c => `<span class="cap-badge cap-${this._escapeHtml(c)}">${this._escapeHtml(c)}</span>`).join('')}</div>`;
                }

                const btnLabel = alreadyAdded ? 'Added' : 'Add';
                item.innerHTML = `
                    <div class="model-list-item-info">
                        <div class="model-list-item-name">${this._escapeHtml(displayName)}</div>
                        <div class="model-list-item-meta">${metaHtml}</div>
                        ${capsHtml}
                    </div>
                    <button class="btn btn-sm ${alreadyAdded ? 'btn-secondary' : 'btn-accent'}" ${alreadyAdded ? 'disabled' : ''}>${btnLabel}</button>
                `;
                if (!alreadyAdded) {
                    item.querySelector('button').addEventListener('click', (e) => {
                        e.stopPropagation();
                        this._addToRegistry(m);
                    });
                }
                this.scannedListEl.appendChild(item);
            }
        } catch (err) {
            this.scannedListEl.innerHTML = '';
            this.state.showToast('Scan failed: ' + err.message, 'error');
        }
    }

    async _addToRegistry(scanned) {
        try {
            const maxCtx = scanned.context_length || 32768;
            await this.api.addModel({
                name: scanned.display_name || scanned.model_name || 'Unknown',
                path: scanned.file_path,
                context_default: Math.min(4096, maxCtx),
                context_recommended: Math.min(8192, maxCtx),
                context_max: maxCtx,
                capabilities: scanned.capabilities || [],
            });
            this.state.showToast('Model added to registry', 'success');
            await this._loadRegistry();
            // Re-render scan list to grey out the newly added model
            await this._scanModels();
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
                this.registryListEl.innerHTML = '<div class="empty-state">No models registered. Scan and add models.</div>';
                this.configPanel.style.display = 'none';
                return;
            }

            for (const m of models) {
                const item = document.createElement('div');
                item.className = 'model-list-item' + (m.id === this._selectedRegistryId ? ' selected' : '');

                // Show rich metadata
                let metaHtml = '';
                metaHtml += `<span class="meta-tag">GPU: ${m.gpu_layers === -1 ? 'auto' : m.gpu_layers}</span>`;
                metaHtml += `<span class="meta-tag">ctx: ${m.context_default.toLocaleString()}</span>`;
                metaHtml += `<span class="meta-tag">T: ${m.temperature}</span>`;
                const notesHtml = m.notes ? `<div class="model-list-item-notes">${this._escapeHtml(m.notes)}</div>` : '';
                let regCapsHtml = '';
                if (m.capabilities && m.capabilities.length) {
                    regCapsHtml = `<div class="model-list-item-caps">${m.capabilities.map(c => `<span class="cap-badge cap-${this._escapeHtml(c)}">${this._escapeHtml(c)}</span>`).join('')}</div>`;
                }

                item.innerHTML = `
                    <div class="model-list-item-info">
                        <div class="model-list-item-name">${this._escapeHtml(m.name)}</div>
                        <div class="model-list-item-meta">${metaHtml}</div>
                        ${regCapsHtml}
                        ${notesHtml}
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

        // Set all slider + input pairs
        this._setSliderInputValue('slider-gpu-layers', 'input-gpu-layers', 'gpu-layers-value', model.gpu_layers);
        this._setSliderInputValue('slider-ctx-length', 'input-ctx-length', 'ctx-length-value', model.context_default);
        this._setSliderInputValue('slider-temp', 'input-temp', 'temp-value', model.temperature);
        this._setSliderInputValue('slider-topp', 'input-topp', 'topp-value', model.top_p);

        // Capabilities and notes
        document.getElementById('model-capabilities').value = (model.capabilities || []).join(', ');
        document.getElementById('model-notes').value = model.notes || '';

        // Update context slider max and add snap points
        if (model.context_max) {
            document.getElementById('slider-ctx-length').max = model.context_max;
            document.getElementById('input-ctx-length').max = model.context_max;
        }
        this._updateSliderMarks(model);

        // Highlight selected
        this.registryListEl.querySelectorAll('.model-list-item').forEach(el => el.classList.remove('selected'));
        const items = this.registryListEl.querySelectorAll('.model-list-item');
        const idx = this.state.registryModels.findIndex(m => m.id === model.id);
        if (idx >= 0 && items[idx]) items[idx].classList.add('selected');

        this._fetchEstimate();
    }

    _setSliderInputValue(sliderId, inputId, valueId, value) {
        document.getElementById(sliderId).value = value;
        document.getElementById(inputId).value = value;
        document.getElementById(valueId).textContent = value;
    }

    async _updateSliderMarks(model) {
        // Context length marks
        const ctxSlider = document.getElementById('slider-ctx-length');
        const ctxMax = model.context_max || 32768;
        const ctxDefault = model.context_default || 4096;
        const ctxRecommended = model.context_recommended || 8192;

        // Build context snap points
        const ctxSnaps = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            .filter(v => v <= ctxMax);
        this._addDatalist('ctx-snaps', ctxSnaps, ctxSlider);

        // GPU layers marks
        try {
            const est = await this.api.recommendLayers(model.path);
            const recommended = est.recommended_gpu_layers || 0;
            const gpuSlider = document.getElementById('slider-gpu-layers');

            // Show recommended mark
            const markEl = document.getElementById('gpu-layers-recommended');
            if (markEl) markEl.remove();
            const mark = document.createElement('div');
            mark.id = 'gpu-layers-recommended';
            mark.className = 'slider-mark';
            mark.textContent = `Recommended: ${recommended}`;
            gpuSlider.parentElement.appendChild(mark);

            // Set max to total layers if we know them
            if (recommended > 0) {
                gpuSlider.max = Math.max(parseInt(gpuSlider.max), recommended + 10);
                document.getElementById('input-gpu-layers').max = gpuSlider.max;
            }
        } catch {}

        // Context recommended mark
        const ctxMarkEl = document.getElementById('ctx-length-recommended');
        if (ctxMarkEl) ctxMarkEl.remove();
        const ctxMark = document.createElement('div');
        ctxMark.id = 'ctx-length-recommended';
        ctxMark.className = 'slider-mark';
        ctxMark.textContent = `Default: ${ctxDefault.toLocaleString()} · Recommended: ${ctxRecommended.toLocaleString()} · Max: ${ctxMax.toLocaleString()}`;
        ctxSlider.parentElement.appendChild(ctxMark);
    }

    _addDatalist(id, values, slider) {
        // Remove old datalist
        const old = document.getElementById(id);
        if (old) old.remove();
        const list = document.createElement('datalist');
        list.id = id;
        for (const v of values) {
            const opt = document.createElement('option');
            opt.value = v;
            list.appendChild(opt);
        }
        slider.parentElement.appendChild(list);
        slider.setAttribute('list', id);
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
            if (est.vram_needed_mb != null) html += `VRAM: ~${est.vram_needed_mb} MB needed`;
            if (est.vram_available_mb != null) html += ` (${est.vram_available_mb} MB free)\n`;
            if (est.ram_needed_mb != null) html += `RAM:  ~${est.ram_needed_mb} MB needed`;
            if (est.ram_available_mb != null) html += ` (${est.ram_available_mb} MB free)\n`;
            if (est.feasible != null) {
                html += est.feasible ? '✓ Should fit' : '✗ May not fit!';
            }
            if (est.warning) html += `\n⚠ ${est.warning}`;
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
                notes: document.getElementById('model-notes').value,
                capabilities: document.getElementById('model-capabilities').value
                    .split(',').map(s => s.trim().toLowerCase()).filter(Boolean),
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
                    this._syncSidebarSliders();
                    this.state.showToast('Model loaded', 'success');
                } else if (status.error) {
                    throw new Error(status.error);
                } else {
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
            const nameEl = document.getElementById('current-model-name');
            nameEl.textContent = 'No model loaded';
            nameEl.classList.remove('loaded');
            document.getElementById('model-status-dot').className = 'status-dot offline';
            document.getElementById('model-status-text').textContent = '';
            document.getElementById('sidebar-model-caps').innerHTML = '';
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
        const dot = document.getElementById('model-status-dot');
        const capsEl = document.getElementById('sidebar-model-caps');
        const statusTextEl = document.getElementById('model-status-text');

        if (status.loading) {
            const pct = Math.round((status.load_progress || 0) * 100);
            nameEl.textContent = status.current_model?.name || 'Loading...';
            nameEl.classList.remove('loaded');
            dot.className = 'status-dot loading';
            statusTextEl.textContent = `Loading... ${pct}%`;
            unloadBtn.disabled = false;
            loadBtn.disabled = true;
            capsEl.innerHTML = '';
        } else if (status.loaded) {
            nameEl.textContent = status.current_model?.name || 'Model loaded';
            nameEl.classList.add('loaded');
            dot.className = 'status-dot online';
            statusTextEl.textContent = 'Ready';
            unloadBtn.disabled = false;
            loadBtn.disabled = false;
            const caps = status.current_model?.capabilities || [];
            capsEl.innerHTML = caps.map(c =>
                `<span class="cap-badge cap-${this._escapeHtml(c)}">${this._escapeHtml(c)}</span>`
            ).join('');
        } else if (status.error) {
            nameEl.textContent = 'No model loaded';
            nameEl.classList.remove('loaded');
            dot.className = 'status-dot error';
            statusTextEl.textContent = status.error.length > 80
                ? status.error.substring(0, 80) + '...'
                : status.error;
            unloadBtn.disabled = true;
            loadBtn.disabled = false;
            capsEl.innerHTML = '';
        } else {
            nameEl.textContent = 'No model loaded';
            nameEl.classList.remove('loaded');
            dot.className = 'status-dot offline';
            statusTextEl.textContent = '';
            unloadBtn.disabled = true;
            loadBtn.disabled = false;
            capsEl.innerHTML = '';
        }
    }

    _syncSidebarSliders() {
        // Populate sidebar inline sliders from the currently selected registry model
        if (!this._selectedRegistryId || !this.state.registryModels) return;
        const model = this.state.registryModels.find(m => m.id === this._selectedRegistryId);
        if (!model) return;
        const setSlider = (id, valId, value) => {
            const s = document.getElementById(id);
            const v = document.getElementById(valId);
            if (s && v) { s.value = value; v.textContent = value; }
        };
        setSlider('sidebar-slider-temp', 'sidebar-temp-val', model.temperature);
        setSlider('sidebar-slider-ctx', 'sidebar-ctx-val', model.context_default);
        setSlider('sidebar-slider-gpu', 'sidebar-gpu-val', model.gpu_layers);
        // Update ctx slider max
        if (model.context_max) {
            const ctxSlider = document.getElementById('sidebar-slider-ctx');
            if (ctxSlider) ctxSlider.max = model.context_max;
        }
    }

    async refreshStatus() {
        try {
            const status = await this.api.getModelStatus();
            this._updateModelDisplay(status);

            // Track loaded model as selected
            if (status.loaded && status.current_model) {
                this._selectedRegistryId = status.current_model.id;
            }

            const models = await this.api.getRegistry();
            this.state.registryModels = models;
            document.getElementById('btn-load-model').disabled = models.length === 0;
            this._syncSidebarSliders();
        } catch {}
    }

    _formatParams(count) {
        if (count >= 1e9) return (count / 1e9).toFixed(1) + 'B';
        if (count >= 1e6) return (count / 1e6).toFixed(0) + 'M';
        return count.toLocaleString();
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
