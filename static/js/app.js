/**
 * NHẬN DẠNG SỐ V2 - Frontend Application Logic
 * Handles: Drag & Drop, File Upload, Clipboard Paste, API calls, Result rendering
 */

// ============================================================
// DOM Elements
// ============================================================
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadBrowse = document.getElementById('upload-browse');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const btnRemove = document.getElementById('btn-remove');
const btnRecognize = document.getElementById('btn-recognize');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const uploadSection = document.getElementById('upload-section');

// Result elements
const summaryTotalValue = document.getElementById('summary-total-value');
const summaryConfidenceValue = document.getElementById('summary-confidence-value');
const summaryAllValue = document.getElementById('summary-all-value');
const resultImage = document.getElementById('result-image');
const resultTableBody = document.getElementById('result-table-body');
const btnDownload = document.getElementById('btn-download');
const btnTryAgain = document.getElementById('btn-try-again');
const btnErrorRetry = document.getElementById('btn-error-retry');
const errorMessage = document.getElementById('error-message');

// State
let selectedFile = null;


// ============================================================
// File Upload Handlers
// ============================================================

/** Open file picker */
uploadBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

/** Handle file selection */
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

// ============================================================
// Drag & Drop
// ============================================================

uploadZone.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// ============================================================
// Clipboard Paste (Ctrl+V)
// ============================================================

document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            if (file) handleFile(file);
            break;
        }
    }
});

// ============================================================
// File Handling
// ============================================================

/**
 * Validate and preview the selected file.
 * @param {File} file
 */
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Định dạng không hỗ trợ', 'Vui lòng chọn file ảnh: JPG, PNG, GIF, BMP, WebP');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File quá lớn', 'Kích thước file tối đa là 16MB. Vui lòng chọn file nhỏ hơn.');
        return;
    }

    selectedFile = file;

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadZone.style.display = 'none';
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

/** Remove selected file and reset */
btnRemove.addEventListener('click', () => {
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    uploadZone.style.display = 'block';
    hideResults();
    hideError();
    hideLoading();
    uploadSection.style.display = 'block';
}

// ============================================================
// OCR Recognition
// ============================================================

btnRecognize.addEventListener('click', async () => {
    if (!selectedFile) return;

    showLoading();
    hideError();
    hideResults();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/api/recognize', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        hideLoading();

        if (data.success) {
            showResults(data);
        } else {
            showError('Không thể nhận dạng', data.error || 'Đã xảy ra lỗi khi xử lý ảnh.');
        }
    } catch (err) {
        hideLoading();
        showError('Lỗi kết nối', 'Không thể kết nối đến server. Vui lòng kiểm tra server đang chạy.');
        console.error('Recognition error:', err);
    }
});

// ============================================================
// Display Results
// ============================================================

/**
 * Render recognition results.
 * @param {Object} data - API response
 */
function showResults(data) {
    const { results, annotated_image, all_numbers, total_found } = data;

    // Summary cards
    summaryTotalValue.textContent = total_found;

    // Average confidence
    if (results.length > 0) {
        const avgConf = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;
        summaryConfidenceValue.textContent = `${Math.round(avgConf * 100)}%`;
    } else {
        summaryConfidenceValue.textContent = '—';
    }

    summaryAllValue.textContent = all_numbers || '—';

    // Annotated image
    if (annotated_image) {
        resultImage.src = annotated_image;
    }

    // Detail table
    resultTableBody.innerHTML = '';
    results.forEach((result, index) => {
        const row = createResultRow(result, index + 1);
        resultTableBody.appendChild(row);
    });

    // Show no-result message if empty
    if (results.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="6" style="text-align: center; padding: 32px; color: var(--text-muted);">
                Không tìm thấy số nào trong ảnh. Hãy thử với ảnh khác.
            </td>
        `;
        resultTableBody.appendChild(row);
    }

    // Animate summary values
    animateValue(summaryTotalValue, 0, total_found, 600);

    resultsSection.style.display = 'block';
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Create a table row for a single result.
 */
function createResultRow(result, index) {
    const row = document.createElement('tr');

    const confPercent = Math.round(result.confidence * 100);
    const confLevel = confPercent >= 80 ? 'high' : confPercent >= 50 ? 'medium' : 'low';

    const typeLabels = {
        'number': 'Số',
        'phone': 'SĐT'
    };

    const x = result.bbox[0][0];
    const y = result.bbox[0][1];

    const engine = result.engine || 'MNIST CNN';

    row.innerHTML = `
        <td style="color: var(--text-muted); font-weight: 600;">${index}</td>
        <td><span class="number-text">${escapeHtml(result.text)}</span></td>
        <td><span class="type-badge ${result.type}">${typeLabels[result.type] || result.type}</span></td>
        <td><span class="engine-badge">${escapeHtml(engine)}</span></td>
        <td>
            <div class="confidence-bar-wrapper">
                <div class="confidence-bar">
                    <div class="confidence-bar-fill ${confLevel}" style="width: ${confPercent}%"></div>
                </div>
                <span class="confidence-text ${confLevel}">${confPercent}%</span>
            </div>
        </td>
        <td><span class="position-text">(${x}, ${y})</span></td>
    `;

    // Animate row appearance
    row.style.opacity = '0';
    row.style.transform = 'translateY(10px)';
    row.style.transition = 'all 0.3s ease';
    setTimeout(() => {
        row.style.opacity = '1';
        row.style.transform = 'translateY(0)';
    }, index * 80);

    return row;
}

// ============================================================
// Download Annotated Image
// ============================================================

btnDownload.addEventListener('click', () => {
    const img = resultImage.src;
    if (!img) return;

    const link = document.createElement('a');
    link.download = 'nhan_dang_so_result.jpg';
    link.href = img;
    link.click();
});

// ============================================================
// Try Again
// ============================================================

btnTryAgain.addEventListener('click', resetUpload);
btnErrorRetry.addEventListener('click', resetUpload);

// ============================================================
// UI Helpers
// ============================================================

function showLoading() {
    loadingSection.style.display = 'block';
    previewContainer.style.display = 'none';
    uploadZone.style.display = 'none';
}

function hideLoading() {
    loadingSection.style.display = 'none';
}

function showError(title, message) {
    document.getElementById('error-title').textContent = title;
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    previewContainer.style.display = 'none';
    uploadZone.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Animate a number counting up.
 */
function animateValue(element, start, end, duration) {
    if (end === 0) {
        element.textContent = '0';
        return;
    }

    const startTime = performance.now();
    const animate = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);
        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    };
    requestAnimationFrame(animate);
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// Image Zoom (click to toggle)
// ============================================================

resultImage.addEventListener('click', () => {
    resultImage.classList.toggle('zoomed');
    if (resultImage.classList.contains('zoomed')) {
        resultImage.style.cursor = 'zoom-out';
        resultImage.style.transform = 'scale(1.5)';
    } else {
        resultImage.style.cursor = 'zoom-in';
        resultImage.style.transform = 'scale(1)';
    }
});

// ============================================================
// Init
// ============================================================
console.log('🔢 Nhận Dạng Số V2 - Ready');
