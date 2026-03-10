/**
 * Settings modal — General config and prompt file editors.
 */

export class SettingsUI {
    constructor(api, state) {
        this.api = api;
        this.state = state;

        this.modal = document.getElementById('settings-modal');
        this._bindEvents();
    }

    _bindEvents() {
        // Open
        document.getElementById('btn-settings').addEventListener('click', () => this.open());

        // Close
        this.modal.querySelectorAll('.modal-close, .modal-overlay').forEach(el => {
            el.addEventListener('click', () => this.close());
        });

        // Tabs
        this.modal.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => this._switchTab(tab.dataset.tab));
        });

        // Slider + input sync for VRAM limit
        this._syncSliderInput('setting-vram-limit', 'input-vram-limit', 'vram-limit-value', '%');
        this._syncSliderInput('setting-ram-limit', 'input-ram-limit', 'ram-limit-value', '%');

        // Save settings
        document.getElementById('btn-save-settings').addEventListener('click', () => this._saveSettings());

        // Save prompt buttons
        this.modal.querySelectorAll('.btn-save-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                const filename = btn.dataset.file;
                this._savePrompt(filename, btn);
            });
        });
    }

    _syncSliderInput(sliderId, inputId, valueId, suffix = '') {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        const valueEl = document.getElementById(valueId);

        slider.addEventListener('input', () => {
            const v = parseInt(slider.value);
            input.value = v;
            valueEl.textContent = v + suffix;
        });

        input.addEventListener('input', () => {
            let v = parseInt(input.value);
            const min = parseInt(slider.min);
            const max = parseInt(slider.max);
            if (v < min) v = min;
            if (v > max) v = max;
            slider.value = v;
            valueEl.textContent = v + suffix;
        });

        input.addEventListener('blur', () => {
            let v = parseInt(input.value);
            const min = parseInt(slider.min);
            const max = parseInt(slider.max);
            if (isNaN(v)) v = parseInt(slider.value);
            if (v < min) v = min;
            if (v > max) v = max;
            input.value = v;
            slider.value = v;
            valueEl.textContent = v + suffix;
        });
    }

    async open() {
        this.modal.style.display = 'flex';
        await this._loadSettings();
        this._switchTab('general');
    }

    close() {
        this.modal.style.display = 'none';
    }

    _switchTab(tabName) {
        this.modal.querySelectorAll('.tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });

        this.modal.querySelectorAll('.tab-content').forEach(tc => {
            tc.classList.toggle('active', tc.id === `tab-${tabName}`);
        });

        const promptMap = {
            'system-prompt': 'system-prompt.md',
            'identity': 'identity.md',
            'user': 'user.md',
            'memory': 'memory.md',
        };
        if (promptMap[tabName]) {
            this._loadPrompt(promptMap[tabName], tabName);
        }
    }

    async _loadSettings() {
        try {
            const config = await this.api.getConfig();
            document.getElementById('setting-models-dir').value = config.models_directory || '';
            document.getElementById('setting-load-on-start').checked = !!config.load_model_on_start;

            const vramLimit = config.vram_limit_percent || 95;
            document.getElementById('setting-vram-limit').value = vramLimit;
            document.getElementById('input-vram-limit').value = vramLimit;
            document.getElementById('vram-limit-value').textContent = vramLimit + '%';

            const ramLimit = config.ram_limit_percent || 85;
            document.getElementById('setting-ram-limit').value = ramLimit;
            document.getElementById('input-ram-limit').value = ramLimit;
            document.getElementById('ram-limit-value').textContent = ramLimit + '%';

            // Populate default model dropdown
            const select = document.getElementById('setting-default-model');
            select.innerHTML = '<option value="">None</option>';
            if (this.state.registryModels) {
                for (const m of this.state.registryModels) {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name;
                    if (m.id === config.default_model_id) opt.selected = true;
                    select.appendChild(opt);
                }
            }
        } catch (err) {
            this.state.showToast('Failed to load settings: ' + err.message, 'error');
        }
    }

    async _saveSettings() {
        const indicator = document.getElementById('settings-save-indicator');
        try {
            await this.api.updateConfig({
                models_directory: document.getElementById('setting-models-dir').value,
                default_model_id: document.getElementById('setting-default-model').value || null,
                load_model_on_start: document.getElementById('setting-load-on-start').checked,
                vram_limit_percent: parseInt(document.getElementById('setting-vram-limit').value),
                ram_limit_percent: parseInt(document.getElementById('setting-ram-limit').value),
            });
            indicator.textContent = 'Saved';
            indicator.classList.add('show');
            setTimeout(() => indicator.classList.remove('show'), 2000);
        } catch (err) {
            this.state.showToast('Save failed: ' + err.message, 'error');
        }
    }

    async _loadPrompt(filename, tabName) {
        const textareaId = `prompt-${tabName}`;
        const textarea = document.getElementById(textareaId);
        if (!textarea) return;

        try {
            const data = await this.api.getPromptFile(filename);
            textarea.value = data.content || '';
        } catch {
            textarea.value = '';
        }
    }

    async _savePrompt(filename, btn) {
        const tabName = filename.replace('.md', '');
        const textarea = document.getElementById(`prompt-${tabName}`);
        if (!textarea) return;

        const indicator = btn.nextElementSibling;

        try {
            await this.api.savePromptFile(filename, textarea.value);
            if (indicator) {
                indicator.textContent = 'Saved';
                indicator.classList.add('show');
                setTimeout(() => indicator.classList.remove('show'), 2000);
            }
        } catch (err) {
            this.state.showToast('Save failed: ' + err.message, 'error');
        }
    }
}
