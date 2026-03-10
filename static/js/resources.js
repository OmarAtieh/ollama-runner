/**
 * Resource monitoring — polls system resources and updates sidebar bars.
 */

export class ResourceMonitor {
    constructor(api) {
        this.api = api;
        this._interval = null;
        this._pollMs = 2000;

        this.cpuBar = document.getElementById('cpu-bar');
        this.cpuText = document.getElementById('cpu-text');
        this.gpuBar = document.getElementById('gpu-bar');
        this.gpuText = document.getElementById('gpu-text');
        this.vramBar = document.getElementById('vram-bar');
        this.vramText = document.getElementById('vram-text');
        this.ramBar = document.getElementById('ram-bar');
        this.ramText = document.getElementById('ram-text');
        this.gpuTempEl = document.getElementById('gpu-temp');

        this._bindVisibility();
    }

    start() {
        if (this._interval) return;
        this._poll(); // immediate first poll
        this._interval = setInterval(() => this._poll(), this._pollMs);
    }

    stop() {
        if (this._interval) {
            clearInterval(this._interval);
            this._interval = null;
        }
    }

    _bindVisibility() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.stop();
            } else {
                this.start();
            }
        });
    }

    async _poll() {
        try {
            const r = await this.api.getResources();
            this._update(r);
        } catch {
            // Silently fail — will retry next poll
        }
    }

    _update(r) {
        // CPU
        const cpuPct = r.cpu_percent || 0;
        this._setBar(this.cpuBar, cpuPct);
        this.cpuText.textContent = cpuPct.toFixed(0) + '%';

        // GPU utilization
        const gpuPct = r.gpu_utilization || 0;
        this._setBar(this.gpuBar, gpuPct);
        this.gpuText.textContent = gpuPct.toFixed(0) + '%';

        // VRAM
        const vramUsed = r.gpu_vram_used_mb || 0;
        const vramTotal = r.gpu_vram_total_mb || 0;
        const vramPct = vramTotal > 0 ? (vramUsed / vramTotal) * 100 : 0;
        this._setBar(this.vramBar, vramPct);
        this.vramText.textContent = vramTotal > 0
            ? `${(vramUsed / 1024).toFixed(1)}/${(vramTotal / 1024).toFixed(1)}G`
            : '--';

        // RAM
        const ramUsed = r.ram_used_mb || 0;
        const ramTotal = r.ram_total_mb || 0;
        const ramPct = ramTotal > 0 ? (ramUsed / ramTotal) * 100 : 0;
        this._setBar(this.ramBar, ramPct);
        this.ramText.textContent = ramTotal > 0
            ? `${(ramUsed / 1024).toFixed(1)}/${(ramTotal / 1024).toFixed(1)}G`
            : '--';

        // GPU temp
        if (r.gpu_temperature != null && r.gpu_temperature > 0) {
            this.gpuTempEl.style.display = 'block';
            this.gpuTempEl.textContent = `Temp: ${r.gpu_temperature}\u00B0C`;
        } else {
            this.gpuTempEl.style.display = 'none';
        }
    }

    _setBar(barEl, pct) {
        pct = Math.max(0, Math.min(100, pct));
        barEl.style.width = pct + '%';
        barEl.classList.remove('warn', 'danger');
        if (pct > 85) {
            barEl.classList.add('danger');
        } else if (pct > 70) {
            barEl.classList.add('warn');
        }
    }
}
