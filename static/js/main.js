// static/js/main.js

// --- НОВАЯ ЛОГИКА: Переключение между загрузкой файла и URL ---
document.addEventListener('DOMContentLoaded', function() {
    // Переключатель способа загрузки
    const uploadMethodBtns = document.querySelectorAll('.upload-method-btn');
    const uploadForm = document.getElementById('uploadForm');
    const uploadUrlForm = document.getElementById('uploadUrlForm');
    
    uploadMethodBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const method = this.getAttribute('data-method');
            
            // Обновляем активные кнопки
            uploadMethodBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Показываем соответствующую форму
            if (method === 'file') {
                uploadForm.style.display = 'block';
                uploadForm.classList.add('active');
                uploadUrlForm.style.display = 'none';
                uploadUrlForm.classList.remove('active');
            } else {
                uploadForm.style.display = 'none';
                uploadForm.classList.remove('active');
                uploadUrlForm.style.display = 'block';
                uploadUrlForm.classList.add('active');
            }
        });
    });

    // --- НОВАЯ ЛОГИКА: Переключение визуального состояния кнопок ---
    const imageRadio = document.querySelector('input[name="file_type"][value="image"]');
    const videoRadio = document.querySelector('input[name="file_type"][value="video"]');
    const imageLabel = imageRadio ? imageRadio.parentElement : null;
    const videoLabel = videoRadio ? videoRadio.parentElement : null;

    // Функция для обновления активной кнопки
    function updateActiveButton() {
        // Сначала убираем класс у обеих
        imageLabel.classList.remove('active-type-btn');
        videoLabel.classList.remove('active-type-btn');

        // Потом добавляем класс активной
        if (imageRadio.checked) {
            imageLabel.classList.add('active-type-btn');
        } else if (videoRadio.checked) {
            videoLabel.classList.add('active-type-btn');
        }
    }

    // Назначаем обработчики событий
    imageRadio.addEventListener('change', updateActiveButton);
    videoRadio.addEventListener('change', updateActiveButton);

    // Инициализируем состояние при загрузке страницы
    if (imageRadio && videoRadio) {
        updateActiveButton();
    }

    // --- ЛОГИКА ДЛЯ ФОРМЫ URL ---
    const urlInput = document.getElementById('fileUrl');
    const urlInfo = document.getElementById('urlInfo');
    const clearUrlBtn = document.getElementById('clearUrl');

    // Управление ползунком порога для URL формы
    const thresholdSliderUrl = document.getElementById('thresholdUrl');
    const thresholdValueUrl = document.getElementById('thresholdValueUrl');
    if (thresholdSliderUrl && thresholdValueUrl) {
        thresholdValueUrl.textContent = parseFloat(thresholdSliderUrl.value).toFixed(2);
        thresholdSliderUrl.addEventListener('input', function() {
            thresholdValueUrl.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    // Обработка ввода URL с предпросмотром
    if (urlInput && urlInfo && clearUrlBtn) {
        const urlPreview = document.getElementById('urlPreview');
        const urlPreviewImage = document.getElementById('urlPreviewImage');
        
        urlInput.addEventListener('input', function() {
            const url = this.value.trim();
            if (url) {
                urlInfo.textContent = `URL: ${url.length > 50 ? url.substring(0, 50) + '...' : url}`;
                clearUrlBtn.style.display = 'block';
                
                // Пытаемся загрузить предпросмотр
                loadUrlPreview(url);
            } else {
                urlInfo.textContent = 'URL не введен';
                clearUrlBtn.style.display = 'none';
                hideUrlPreview();
            }
        });

        clearUrlBtn.addEventListener('click', function() {
            urlInput.value = '';
            urlInfo.textContent = 'URL не введен';
            clearUrlBtn.style.display = 'none';
            hideUrlPreview();
        });
        
        // Кнопка очистки превью URL
        const clearUrlPreviewBtn = document.getElementById('clearUrlPreview');
        if (clearUrlPreviewBtn) {
            clearUrlPreviewBtn.addEventListener('click', function() {
                hideUrlPreview();
            });
        }
        
        // Функция загрузки предпросмотра
        function loadUrlPreview(url) {
            // Проверяем, что это валидный URL
            try {
                new URL(url);
            } catch (e) {
                hideUrlPreview();
                return;
            }
            
            // Определяем тип по расширению (только изображения)
            const urlLower = url.toLowerCase();
            const isImage = urlLower.match(/\.(jpg|jpeg|png|gif|webp)(\?|$)/i);
            
            if (isImage && urlPreview && urlPreviewImage) {
                urlPreviewImage.src = url;
                urlPreviewImage.style.display = 'block';
                urlPreview.style.display = 'block';
                
                // Обработка ошибок загрузки
                urlPreviewImage.onerror = function() {
                    hideUrlPreview();
                };
            } else {
                hideUrlPreview();
            }
        }
        
        function hideUrlPreview() {
            if (urlPreview) urlPreview.style.display = 'none';
            if (urlPreviewImage) {
                urlPreviewImage.src = '';
                urlPreviewImage.style.display = 'none';
            }
        }
    }

    // Управление ползунком порога уверенности
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('thresholdValue');
    if (thresholdSlider && thresholdValue) {
        thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
        thresholdSlider.addEventListener('input', function() {
            thresholdValue.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    // Управление видимостью ползунка кадров и режима скорости
    const framesSliderSection = document.getElementById('framesSliderSection');
    const framesCountSlider = document.getElementById('framesCount');
    const framesCountValue = document.getElementById('framesCountValue');

    if (framesCountValue && framesCountSlider) {
        framesCountValue.textContent = framesCountSlider.value;
        framesCountSlider.addEventListener('input', function() {
            framesCountValue.textContent = this.value;
        });
    }

    function toggleFramesSlider() {
        if (videoRadio.checked) {
            framesSliderSection.style.display = 'block';
        } else {
            framesSliderSection.style.display = 'none';
        }
    }

    imageRadio.addEventListener('change', toggleFramesSlider);
    videoRadio.addEventListener('change', toggleFramesSlider);
    toggleFramesSlider();
});
// --- КОНЕЦ НОВОЙ ЛОГИКИ ---

// --- НОВАЯ ЛОГИКА: Drag & Drop ---
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.querySelector('.drop-zone');
    const fileInput = document.getElementById('file');
    const fileInfo = document.getElementById('fileInfo');
    const uploadForm = document.getElementById('uploadForm');

    // Предотвращаем стандартные действия браузера при перетаскивании
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Добавляем/удаляем класс при наведении файла
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Обработка сброса файла
    dropZone.addEventListener('drop', handleDrop, false);

    // Обработка вставки из буфера обмена
    document.addEventListener('paste', function(e) {
        // Проверяем, что форма загрузки файла активна
        if (uploadForm && uploadForm.classList.contains('active')) {
            const items = e.clipboardData.items;
            
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                
                // Проверяем, что это изображение
                if (item.type.indexOf('image') !== -1) {
                    const blob = item.getAsFile();
                    const file = new File([blob], 'pasted-image.png', { type: item.type });
                    
                    // Создаем DataTransfer для установки файла в input
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                    
                    // Вызываем событие изменения
                    fileInput.dispatchEvent(new Event('change'));
                    
                    // Устанавливаем тип файла на "Изображение"
                    if (imageRadio) {
                        imageRadio.checked = true;
                        imageRadio.dispatchEvent(new Event('change'));
                    }
                    
                    e.preventDefault();
                    break;
                }
            }
        }
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropZone.classList.add('active');
    }

    function unhighlight(e) {
        dropZone.classList.remove('active');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length) {
            // Симулируем выбор файла в input
            fileInput.files = files;
            // Вызываем событие изменения, чтобы обновить информацию о файле
            fileInput.dispatchEvent(new Event('change'));
        }
    }

    // Обновляем информацию о файле при выборе (и при drag & drop)
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Валидация размера файла (100 МБ)
            const maxSize = 100 * 1024 * 1024; // 100 МБ в байтах
            if (file.size > maxSize) {
                alert(`Файл слишком большой. Максимальный размер: 100 МБ. Ваш файл: ${(file.size / (1024 * 1024)).toFixed(2)} МБ`);
                fileInput.value = '';
                fileInfo.textContent = 'Файл не выбран';
                hidePreview();
                return;
            }
            
            fileInfo.textContent = `Выбран файл: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} МБ)`;
            showPreview(file);
        } else {
            fileInfo.textContent = 'Файл не выбран';
            hidePreview();
        }
    });

    // Функция показа превью
    function showPreview(file) {
        const preview = document.getElementById('filePreview');
        const previewImage = document.getElementById('previewImage');
        const previewVideo = document.getElementById('previewVideo');
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                previewVideo.style.display = 'none';
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else if (file.type.startsWith('video/')) {
            previewVideo.src = URL.createObjectURL(file);
            previewVideo.style.display = 'block';
            previewImage.style.display = 'none';
            preview.style.display = 'block';
        }
    }

    // Функция скрытия превью (делаем глобальной)
    window.hidePreview = function() {
        const preview = document.getElementById('filePreview');
        const previewImage = document.getElementById('previewImage');
        const previewVideo = document.getElementById('previewVideo');
        if (preview) preview.style.display = 'none';
        if (previewImage) previewImage.style.display = 'none';
        if (previewVideo) {
            if (previewVideo.src) {
                URL.revokeObjectURL(previewVideo.src);
                previewVideo.src = '';
            }
            previewVideo.style.display = 'none';
        }
    };

    // Кнопка очистки превью
    const clearPreviewBtn = document.getElementById('clearPreview');
    if (clearPreviewBtn) {
        clearPreviewBtn.addEventListener('click', function() {
            fileInput.value = '';
            fileInfo.textContent = 'Файл не выбран';
            hidePreview();
        });
    }
});
// --- КОНЕЦ НОВОЙ ЛОГИКИ DRAG & DROP ---

// --- НОВАЯ ЛОГИКА: Фокус на результат ---
function scrollToResult() {
    // Используем setTimeout, чтобы убедиться, что DOM обновлён
    setTimeout(() => {
        const resultDiv = document.getElementById('result');
        if (resultDiv && resultDiv.children.length > 0) { // Проверяем, есть ли результаты
            resultDiv.scrollIntoView({ behavior: 'smooth' }); // Плавная прокрутка
        }
    }, 100); // Небольшая задержка для уверенности
}
// --- КОНЕЦ НОВОЙ ЛОГИКИ ---

// --- ФУНКЦИЯ ОТКРЫТИЯ МОДАЛЬНОГО ОКНА ---
function openImageModal(imageSrc, verdict, probability) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalInfo = document.getElementById('modalInfo');
    
    if (modal && modalImage && modalInfo) {
        modalImage.src = imageSrc;
        
        // Формируем содержимое модального окна
        let infoHTML = '';
        if (verdict && verdict !== 'Временная шкала вероятности') {
            // Для обычных изображений лиц
            const isOriginal = verdict.includes('Оригинал');
            infoHTML = `
                <div class="modal-info-verdict ${isOriginal ? 'original' : 'deepfake'}">
                    ${verdict}
                </div>
                ${probability !== null && probability !== undefined ? `
                <div class="modal-info-probability">
                    Вероятность: ${(probability * 100).toFixed(1)}%
                </div>` : ''}
            `;
        } else {
            // Для графика временной шкалы - убираем подсказку, так как график уже открыт
            infoHTML = `
                <div class="modal-info-verdict" style="color: var(--info-color); background: rgba(30, 144, 255, 0.15);">
                    ${verdict || 'График временной шкалы'}
                </div>
            `;
        }
        
        modalInfo.innerHTML = infoHTML;
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Блокируем прокрутку фона
    }
}

// Закрытие модального окна
function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = ''; // Восстанавливаем прокрутку
    }
}

// Обработчики для модального окна
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('imageModal');
    const modalClose = document.getElementById('modalClose');
    const modalOverlay = document.querySelector('.modal-overlay');
    
    if (modalClose) {
        modalClose.addEventListener('click', closeImageModal);
    }
    
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeImageModal);
    }
    
    // Закрытие по Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal && modal.style.display === 'flex') {
            closeImageModal();
        }
    });
});
// --- КОНЕЦ ФУНКЦИИ МОДАЛЬНОГО ОКНА ---

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file');
    const fileTypeInputs = document.getElementsByName('file_type');
    const framesCountInput = document.getElementById('framesCount');
    const thresholdInput = document.getElementById('threshold');
    const speedModeInputs = document.getElementsByName('speed_mode');

    if (fileInput.files.length === 0) {
        alert('Пожалуйста, выберите файл.');
        return;
    }

    const selectedFile = fileInput.files[0];
    formData.append('file', selectedFile);

    let selectedFileType = 'image';
    for (const radio of fileTypeInputs) {
        if (radio.checked) {
            selectedFileType = radio.value;
            break;
        }
    }
    formData.append('file_type', selectedFileType);

    // Добавляем порог уверенности
    if (thresholdInput) {
        formData.append('threshold', parseFloat(thresholdInput.value));
    }

    // Добавляем параметры для видео
    if (selectedFileType === 'video') {
        if (framesCountInput) {
        formData.append('frames_count', framesCountInput.value);
    }
        // Режим скорости
        for (const radio of speedModeInputs) {
            if (radio.checked) {
                formData.append('speed_mode', radio.value);
                break;
            }
        }
    }

    const progressDiv = document.getElementById('progress');
    const progressText = document.getElementById('progressText');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const progressBar = document.getElementById('progressBar');
    const resultDiv = document.getElementById('result');
    const clearResultsBtn = document.getElementById('clearResults');
    
    progressDiv.style.display = 'flex';
    resultDiv.innerHTML = '';
    
    if (clearResultsBtn) {
        clearResultsBtn.style.display = 'none';
    }

    // Обновляем информацию о файле
    document.getElementById('fileInfo').textContent = `Выбран файл: ${selectedFile.name}`;
    
    // Для видео просто показываем текст без прогресс-бара
    if (selectedFileType === 'video') {
        if (progressText) progressText.textContent = 'Обработка видео...';
        if (progressBarContainer) progressBarContainer.style.display = 'none';
    } else {
        if (progressText) progressText.textContent = 'Анализируется...';
        if (progressBarContainer) progressBarContainer.style.display = 'none';
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        return response.json();
    })
    .then(data => {
        progressDiv.style.display = 'none';

        if (data.detail) { // Ошибка в формате FastAPI
            resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Ошибка: ${data.detail}</div>`;
        } else if (data.message) {
            resultDiv.innerHTML = ''; // Очистить перед рендерингом

            // --- ДОБАВЛЕНО: Отображение сообщения ---
            const messageDiv = document.createElement('div');
            messageDiv.className = 'result-message';
            messageDiv.textContent = data.message;
            resultDiv.appendChild(messageDiv);
            // ---


            if (data.result && data.result.annotated_image) {
                // Результат для изображения
                const imgContainer = document.createElement('div');
                imgContainer.className = 'result-item card';

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';

                const imgTitle = document.createElement('h3');
                imgTitle.className = 'card-title';
                imgTitle.textContent = 'Результат анализа изображения';
                cardBody.appendChild(imgTitle);

                const img = document.createElement('img');
                img.src = `/uploads/${encodeURIComponent(data.result.annotated_image)}`;
                img.alt = 'Аннотированное изображение';
                img.className = 'result-image card-img-top'; // Bootstrap class
                cardBody.appendChild(img);

                // Добавляем результаты для каждого лица
                if (data.result.face_results && Array.isArray(data.result.face_results)) {
                    const resultsDiv = document.createElement('div'); // Контейнер для результатов
                    data.result.face_results.forEach(faceRes => {
                        if (faceRes.error) {
                             const errorDiv = document.createElement('div');
                             errorDiv.className = 'result-text warning'; // Используем warning для ошибок
                             errorDiv.innerHTML = `<strong>Ошибка:</strong> ${faceRes.error}`;
                             resultsDiv.appendChild(errorDiv);
                        } else {
                            const verdictClass = faceRes.prediction.includes('Оригинал') ? 'success' : 'danger';
                            const isOriginal = faceRes.prediction.includes('Оригинал');
                            const badgeIcon = isOriginal ? '✓' : '✗';
                            const badgeText = isOriginal ? 'Оригинал' : 'Дипфейк';
                            
                            const faceDiv = document.createElement('div');
                            faceDiv.className = `result-text ${verdictClass}`;
                            faceDiv.innerHTML = `
                                <span class="verdict-badge ${verdictClass}">${badgeIcon} ${badgeText}</span>
                                <strong>Лицо ${faceRes.face_index}:&nbsp;</strong><span class="verdict-inline">${faceRes.prediction} (Вероятность: ${(faceRes.probability * 100).toFixed(2)}%)</span>
                            `;
                            resultsDiv.appendChild(faceDiv);

                            // --- ВЕРТИКАЛЬНЫЙ БЛОК ДЛЯ КНОПКИ/ТЕПЛОКАРТЫ/КРОПА ---
                            const faceBlock = document.createElement('div');
                            faceBlock.className = 'face-block';

                            // КРОП ЛИЦА (сначала кроп)
                            if (faceRes.face_crop_image) {
                                const faceCropImg = document.createElement('img');
                                faceCropImg.src = `/uploads/${encodeURIComponent(faceRes.face_crop_image)}`;
                                faceCropImg.alt = `Кроп лица ${faceRes.face_index}`;
                                faceCropImg.className = 'face-crop-thumb';
                                faceCropImg.style.cursor = 'pointer';
                                faceCropImg.style.border = `3px solid ${faceRes.prediction.includes('Оригинал') ? 'green' : 'red'}`;
                                faceCropImg.style.borderRadius = '5px';
                                faceCropImg.addEventListener('click', function() {
                                    openImageModal(faceCropImg.src, faceRes.prediction, faceRes.probability);
                                });
                                faceBlock.appendChild(faceCropImg);
                            }

                            // ТЕПЛОВАЯ КАРТА (только для дипфейков) — после кропа
                            if (!faceRes.prediction.includes('Оригинал') && faceRes.heatmap) {
                                const toggleBtn = document.createElement('button');
                                toggleBtn.type = 'button';
                                toggleBtn.className = 'heatmap-toggle-btn';
                                toggleBtn.textContent = 'Показать теплокарту';

                                const heatmapContainer = document.createElement('div');
                                heatmapContainer.className = 'heatmap-container';
                                heatmapContainer.style.display = 'none';

                                const heatmapImg = document.createElement('img');
                                heatmapImg.src = `/uploads/${encodeURIComponent(faceRes.heatmap)}`;
                                heatmapImg.alt = `Тепловая карта лица ${faceRes.face_index}`;
                                heatmapImg.className = 'heatmap-thumb';
                                heatmapImg.style.cursor = 'pointer';
                                heatmapImg.addEventListener('click', function() {
                                    openImageModal(heatmapImg.src, 'Теплокарта (Grad-CAM)', null);
                                });

                                const legend = document.createElement('div');
                                legend.className = 'heatmap-legend';
                                legend.textContent = 'Области повышенного внимания модели; не является объяснением.';

                                heatmapContainer.appendChild(heatmapImg);
                                heatmapContainer.appendChild(legend);

                                toggleBtn.addEventListener('click', function() {
                                    const isHidden = heatmapContainer.style.display === 'none';
                                    heatmapContainer.style.display = isHidden ? 'block' : 'none';
                                    toggleBtn.textContent = isHidden ? 'Скрыть теплокарту' : 'Показать теплокарту';
                                });

                                faceBlock.appendChild(toggleBtn);
                                faceBlock.appendChild(heatmapContainer);
                            }

                            if (faceBlock.children.length > 0) {
                                resultsDiv.appendChild(faceBlock);
                            }
                            // ---
                        }
                    });
                    cardBody.appendChild(resultsDiv);
                }
                imgContainer.appendChild(cardBody);
                resultDiv.appendChild(imgContainer);

            } else if (data.result_video && data.result_video.annotated_video) {
                // Результат для видео
                const videoContainer = document.createElement('div');
                videoContainer.className = 'result-item card';

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';

                const videoTitle = document.createElement('h3');
                videoTitle.className = 'card-title';
                videoTitle.textContent = 'Результат анализа видео';
                cardBody.appendChild(videoTitle);

                // Визуальный индикатор статуса для видео
                const summaryDiv = document.createElement('div');
                const isOriginal = data.result_video.summary.includes('Оригинал');
                const summaryClass = isOriginal ? 'success' : 'danger';
                const badgeIcon = isOriginal ? '✓' : '✗';
                const badgeText = isOriginal ? 'Оригинал' : 'Дипфейк';
                
                summaryDiv.className = `result-text ${summaryClass}`;
                summaryDiv.innerHTML = `
                    <span class="verdict-badge ${summaryClass}">${badgeIcon} ${badgeText}</span>
                    <span>${data.result_video.summary}</span>
                `;
                cardBody.appendChild(summaryDiv);

                // --- ПОКА УБРАНО: Видео ---
                // const video = document.createElement('video');
                // video.src = `/uploads/${encodeURIComponent(data.result_video.annotated_video)}`;
                // video.controls = true;
                // video.className = 'result-video card-img-top'; // Bootstrap class
                // cardBody.appendChild(video);
                // ---

                // --- ОБНОВЛЕНО: Отображение обнаруженных лиц ---
                if (data.result_video.detected_faces && data.result_video.detected_faces.length > 0) {
                    const facesTitle = document.createElement('h4');
                    facesTitle.className = 'mt-3';
                    facesTitle.textContent = 'Обнаруженные лица из видео';
                    cardBody.appendChild(facesTitle);

                    const facesContainer = document.createElement('div');
                    // УБРАНО: facesContainer.className = 'detected-faces-container';
                    // facesContainer.style.display = 'flex';
                    // facesContainer.style.flexWrap = 'wrap';
                    // facesContainer.style.gap = '10px';
                    // facesContainer.style.justifyContent = 'center'; // Центрируем
                    facesContainer.className = 'detected-faces-container'; // НАЗНАЧАЕМ НОВЫЙ КЛАСС ДЛЯ СЕТКИ

                    data.result_video.detected_faces.forEach(faceData => { // Используем faceData вместо face_filename
                        const face_filename = faceData.filename; // Извлекаем имя файла
                        const face_verdict = faceData.verdict;    // Извлекаем verdict
                        const face_prob = faceData.probability;  // Извлекаем probability
                        const frame_index = faceData.frame_index; // Извлекаем frame_index

                        // Создаём карточку для лица
                        const faceCard = document.createElement('div');
                        faceCard.className = 'face-card';
                        // Добавляем класс для рамки в зависимости от вердикта
                        if (face_verdict.includes('Оригинал')) {
                            faceCard.classList.add('face-card-border-original');
                        } else {
                            faceCard.classList.add('face-card-border-deepfake');
                        }

                        // Создаём изображение
                        const faceImg = document.createElement('img');
                        faceImg.src = `/uploads/${encodeURIComponent(face_filename)}`;
                        faceImg.alt = `Обнаруженное лицо (Frame ${frame_index})`;
                        faceImg.style.cursor = 'pointer';
                        // Добавляем обработчик клика для модального окна
                        faceImg.addEventListener('click', function() {
                            openImageModal(faceImg.src, face_verdict, face_prob);
                        });
                        faceCard.appendChild(faceImg);

                        // Создаём блок с информацией
                        const faceInfo = document.createElement('div');
                        faceInfo.className = 'face-card-info';

                        // Вердикт
                        const verdictDiv = document.createElement('div');
                        verdictDiv.className = 'face-card-verdict';
                        if (face_verdict.includes('Оригинал')) {
                            verdictDiv.classList.add('original');
                            verdictDiv.textContent = 'Оригинал';
                        } else {
                            verdictDiv.classList.add('deepfake');
                            verdictDiv.textContent = 'Дипфейк';
                        }
                        faceInfo.appendChild(verdictDiv);

                        // Вероятность
                        const probDiv = document.createElement('div');
                        probDiv.className = 'face-card-probability';
                        probDiv.textContent = `${(face_prob * 100).toFixed(1)}%`;
                        faceInfo.appendChild(probDiv);

                        faceCard.appendChild(faceInfo);
                        facesContainer.appendChild(faceCard);
                    });

                    cardBody.appendChild(facesContainer);
                }
                // ---

                // --- ОТОБРАЖЕНИЕ ВРЕМЕННОЙ ШКАЛЫ ---
                if (data.result_video.plot) {
                    const plotTitle = document.createElement('h4');
                    plotTitle.className = 'mt-3';
                    plotTitle.textContent = '📊 Временная шкала вероятности';
                    cardBody.appendChild(plotTitle);

                    const plotContainer = document.createElement('div');
                    plotContainer.className = 'plot-container';

                    const plotImg = document.createElement('img');
                    plotImg.src = `/uploads/${encodeURIComponent(data.result_video.plot)}`;
                    plotImg.alt = 'Временная шкала вероятности';
                    plotImg.className = 'result-plot';
                    plotImg.style.cursor = 'pointer';
                    
                    // Добавляем обработчик клика для модального окна
                    plotImg.addEventListener('click', function() {
                        openImageModal(plotImg.src, 'Временная шкала вероятности', null);
                    });
                    
                    plotContainer.appendChild(plotImg);
                    cardBody.appendChild(plotContainer);
                }
                // ---

                videoContainer.appendChild(cardBody);
                resultDiv.appendChild(videoContainer);
            }

            // --- ВЫЗОВ ФОКУСА НА РЕЗУЛЬТАТ ---
            scrollToResult();
            
            // Показываем кнопку очистки
            if (clearResultsBtn) {
                clearResultsBtn.style.display = 'block';
            }
            // ---

        } else {
            resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Неизвестный формат ответа от сервера.</div>`;
        }
    })
    .catch(error => {
        // Очищаем интервал прогресса при ошибке
        if (window.videoProgressInterval) {
            clearInterval(window.videoProgressInterval);
            window.videoProgressInterval = null;
        }
        
        progressDiv.style.display = 'none';
        if (progressBarContainer) progressBarContainer.style.display = 'none';
        if (progressBar) progressBar.style.width = '0%';
        
        console.error('Ошибка:', error);
        resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Произошла ошибка при отправке запроса: ${error.message}</div>`;
    });
});

// --- ОБРАБОТЧИК ФОРМЫ ЗАГРУЗКИ ПО URL ---
document.getElementById('uploadUrlForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const urlInput = document.getElementById('fileUrl');
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('Пожалуйста, введите URL.');
        return;
    }

    // Валидация URL
    try {
        new URL(url);
    } catch (e) {
        alert('Некорректный URL. Пожалуйста, введите полный URL (например: https://example.com/image.jpg)');
        return;
    }

    const thresholdInputUrl = document.getElementById('thresholdUrl');

    const formData = new FormData();
    formData.append('url', url);
    formData.append('file_type', 'image'); // Только изображения по URL
    
    // Добавляем порог уверенности
    if (thresholdInputUrl) {
        formData.append('threshold', parseFloat(thresholdInputUrl.value));
    }

    const progressDiv = document.getElementById('progress');
    const progressText = document.getElementById('progressText');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const progressBar = document.getElementById('progressBar');
    const resultDiv = document.getElementById('result');
    const clearResultsBtn = document.getElementById('clearResults');
    
    progressDiv.style.display = 'flex';
    resultDiv.innerHTML = '';
    
    if (clearResultsBtn) {
        clearResultsBtn.style.display = 'none';
    }
    
    // Скрыть превью URL при отправке
    const urlPreview = document.getElementById('urlPreview');
    if (urlPreview) {
        urlPreview.style.display = 'none';
    }
    
    if (progressText) progressText.textContent = 'Анализируется...';
    if (progressBarContainer) progressBarContainer.style.display = 'none';

    fetch('/upload-url', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        progressDiv.style.display = 'none';

        if (data.detail) {
            resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Ошибка: ${data.detail}</div>`;
        } else if (data.message) {
            resultDiv.innerHTML = '';

            const messageDiv = document.createElement('div');
            messageDiv.className = 'result-message';
            messageDiv.textContent = data.message;
            resultDiv.appendChild(messageDiv);

            // Используем ту же логику отображения результатов, что и для обычной загрузки
            if (data.result && data.result.annotated_image) {
                // Результат для изображения
                const imgContainer = document.createElement('div');
                imgContainer.className = 'result-item card';

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';

                const imgTitle = document.createElement('h3');
                imgTitle.className = 'card-title';
                imgTitle.textContent = 'Результат анализа изображения';
                cardBody.appendChild(imgTitle);

                const img = document.createElement('img');
                img.src = `/uploads/${encodeURIComponent(data.result.annotated_image)}`;
                img.alt = 'Аннотированное изображение';
                img.className = 'result-image card-img-top'; // Bootstrap class
                cardBody.appendChild(img);

                // Добавляем результаты для каждого лица
                if (data.result.face_results && Array.isArray(data.result.face_results)) {
                    const resultsDiv = document.createElement('div'); // Контейнер для результатов
                    data.result.face_results.forEach(faceRes => {
                        if (faceRes.error) {
                             const errorDiv = document.createElement('div');
                             errorDiv.className = 'result-text warning'; // Используем warning для ошибок
                             errorDiv.innerHTML = `<strong>Ошибка:</strong> ${faceRes.error}`;
                             resultsDiv.appendChild(errorDiv);
                        } else {
                            const verdictClass = faceRes.prediction.includes('Оригинал') ? 'success' : 'danger';
                            const isOriginal = faceRes.prediction.includes('Оригинал');
                            const badgeIcon = isOriginal ? '✓' : '✗';
                            const badgeText = isOriginal ? 'Оригинал' : 'Дипфейк';

                            const faceDiv = document.createElement('div');
                            faceDiv.className = `result-text ${verdictClass}`;
                            faceDiv.innerHTML = `
                                <span class="verdict-badge ${verdictClass}">${badgeIcon} ${badgeText}</span>
                                <strong>Лицо ${faceRes.face_index}:&nbsp;</strong><span class="verdict-inline">${faceRes.prediction} (Вероятность: ${(faceRes.probability * 100).toFixed(2)}%)</span>
                            `;
                            resultsDiv.appendChild(faceDiv);

                            // --- ВЕРТИКАЛЬНЫЙ БЛОК ДЛЯ КНОПКИ/ТЕПЛОКАРТЫ/КРОПА ---
                            const faceBlock = document.createElement('div');
                            faceBlock.className = 'face-block';

                            // КРОП ЛИЦА (сначала кроп)
                            if (faceRes.face_crop_image) {
                                const faceCropImg = document.createElement('img');
                                faceCropImg.src = `/uploads/${encodeURIComponent(faceRes.face_crop_image)}`;
                                faceCropImg.alt = `Кроп лица ${faceRes.face_index}`;
                                faceCropImg.className = 'face-crop-thumb';
                                faceCropImg.style.cursor = 'pointer';
                                faceCropImg.style.border = `3px solid ${faceRes.prediction.includes('Оригинал') ? 'green' : 'red'}`;
                                faceCropImg.style.borderRadius = '5px';
                                faceCropImg.addEventListener('click', function() {
                                    openImageModal(faceCropImg.src, faceRes.prediction, faceRes.probability);
                                });
                                faceBlock.appendChild(faceCropImg);
                            }

                            // ТЕПЛОВАЯ КАРТА (только для дипфейков) — после кропа
                            if (!faceRes.prediction.includes('Оригинал') && faceRes.heatmap) {
                                const toggleBtn = document.createElement('button');
                                toggleBtn.type = 'button';
                                toggleBtn.className = 'heatmap-toggle-btn';
                                toggleBtn.textContent = 'Показать теплокарту';

                                const heatmapContainer = document.createElement('div');
                                heatmapContainer.className = 'heatmap-container';
                                heatmapContainer.style.display = 'none';

                                const heatmapImg = document.createElement('img');
                                heatmapImg.src = `/uploads/${encodeURIComponent(faceRes.heatmap)}`;
                                heatmapImg.alt = `Тепловая карта лица ${faceRes.face_index}`;
                                heatmapImg.className = 'heatmap-thumb';
                                heatmapImg.style.cursor = 'pointer';
                                heatmapImg.addEventListener('click', function() {
                                    openImageModal(heatmapImg.src, 'Теплокарта (Grad-CAM)', null);
                                });

                                const legend = document.createElement('div');
                                legend.className = 'heatmap-legend';
                                legend.textContent = 'Области повышенного внимания модели; не является объяснением.';

                                heatmapContainer.appendChild(heatmapImg);
                                heatmapContainer.appendChild(legend);

                                toggleBtn.addEventListener('click', function() {
                                    const isHidden = heatmapContainer.style.display === 'none';
                                    heatmapContainer.style.display = isHidden ? 'block' : 'none';
                                    toggleBtn.textContent = isHidden ? 'Скрыть теплокарту' : 'Показать теплокарту';
                                });

                                faceBlock.appendChild(toggleBtn);
                                faceBlock.appendChild(heatmapContainer);
                            }

                            if (faceBlock.children.length > 0) {
                                resultsDiv.appendChild(faceBlock);
                            }
                            // ---
                        }
                    });
                    cardBody.appendChild(resultsDiv);
                }
                imgContainer.appendChild(cardBody);
                resultDiv.appendChild(imgContainer);


            } else if (data.result_video && data.result_video.annotated_video) {
                // Результат для видео - используем ту же функцию рендеринга
                const videoContainer = document.createElement('div');
                videoContainer.className = 'result-item card';

                const cardBody = document.createElement('div');
                cardBody.className = 'card-body';

                const videoTitle = document.createElement('h3');
                videoTitle.className = 'card-title';
                videoTitle.textContent = 'Результат анализа видео';
                cardBody.appendChild(videoTitle);

                // Визуальный индикатор статуса для видео
                const summaryDiv = document.createElement('div');
                const isOriginal = data.result_video.summary.includes('Оригинал');
                const summaryClass = isOriginal ? 'success' : 'danger';
                const badgeIcon = isOriginal ? '✓' : '✗';
                const badgeText = isOriginal ? 'Оригинал' : 'Дипфейк';
                
                summaryDiv.className = `result-text ${summaryClass}`;
                summaryDiv.innerHTML = `
                    <span class="verdict-badge ${summaryClass}">${badgeIcon} ${badgeText}</span>
                    <span>${data.result_video.summary}</span>
                `;
                cardBody.appendChild(summaryDiv);

                if (data.result_video.detected_faces && data.result_video.detected_faces.length > 0) {
                    const facesTitle = document.createElement('h4');
                    facesTitle.className = 'mt-3';
                    facesTitle.textContent = 'Обнаруженные лица из видео';
                    cardBody.appendChild(facesTitle);

                    const facesContainer = document.createElement('div');
                    facesContainer.className = 'detected-faces-container';

                    data.result_video.detected_faces.forEach(faceData => {
                        const face_filename = faceData.filename;
                        const face_verdict = faceData.verdict;
                        const face_prob = faceData.probability;
                        const frame_index = faceData.frame_index;

                        const faceCard = document.createElement('div');
                        faceCard.className = 'face-card';
                        if (face_verdict.includes('Оригинал')) {
                            faceCard.classList.add('face-card-border-original');
                        } else {
                            faceCard.classList.add('face-card-border-deepfake');
                        }

                        const faceImg = document.createElement('img');
                        faceImg.src = `/uploads/${encodeURIComponent(face_filename)}`;
                        faceImg.alt = `Обнаруженное лицо (Frame ${frame_index})`;
                        faceImg.style.cursor = 'pointer';
                        faceImg.addEventListener('click', function() {
                            openImageModal(faceImg.src, face_verdict, face_prob);
                        });
                        faceCard.appendChild(faceImg);

                        const faceInfo = document.createElement('div');
                        faceInfo.className = 'face-card-info';

                        const verdictDiv = document.createElement('div');
                        verdictDiv.className = 'face-card-verdict';
                        if (face_verdict.includes('Оригинал')) {
                            verdictDiv.classList.add('original');
                            verdictDiv.textContent = 'Оригинал';
                        } else {
                            verdictDiv.classList.add('deepfake');
                            verdictDiv.textContent = 'Дипфейк';
                        }
                        faceInfo.appendChild(verdictDiv);

                        const probDiv = document.createElement('div');
                        probDiv.className = 'face-card-probability';
                        probDiv.textContent = `${(face_prob * 100).toFixed(1)}%`;
                        faceInfo.appendChild(probDiv);

                        faceCard.appendChild(faceInfo);
                        facesContainer.appendChild(faceCard);
                    });

                    cardBody.appendChild(facesContainer);
                }

                if (data.result_video.plot) {
                    const plotTitle = document.createElement('h4');
                    plotTitle.className = 'mt-3';
                    plotTitle.textContent = '📊 Временная шкала вероятности';
                    cardBody.appendChild(plotTitle);

                    const plotContainer = document.createElement('div');
                    plotContainer.className = 'plot-container';

                    const plotImg = document.createElement('img');
                    plotImg.src = `/uploads/${encodeURIComponent(data.result_video.plot)}`;
                    plotImg.alt = 'Временная шкала вероятности';
                    plotImg.className = 'result-plot';
                    plotImg.style.cursor = 'pointer';
                    plotImg.addEventListener('click', function() {
                        openImageModal(plotImg.src, 'Временная шкала вероятности', null);
                    });
                    
                    plotContainer.appendChild(plotImg);
                    cardBody.appendChild(plotContainer);
                }

                videoContainer.appendChild(cardBody);
                resultDiv.appendChild(videoContainer);
            }

            scrollToResult();
            
            if (clearResultsBtn) {
                clearResultsBtn.style.display = 'block';
            }
        } else {
            resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Неизвестный формат ответа от сервера.</div>`;
        }
    })
    .catch(error => {
        progressDiv.style.display = 'none';
        console.error('Ошибка:', error);
        resultDiv.innerHTML = `<div class="error-message alert alert-danger" role="alert">Произошла ошибка при отправке запроса: ${error.message}</div>`;
    });
});
// --- КОНЕЦ ОБРАБОТЧИКА ФОРМЫ URL ---

// --- Убедиться, что прогресс скрыт при загрузке страницы ---
document.addEventListener('DOMContentLoaded', function() {
    const progressDiv = document.getElementById('progress');
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }

    // Информационная панель - клик на весь заголовок открывает/закрывает
    const infoPanelHeader = document.querySelector('.info-panel-header');
    const infoToggle = document.getElementById('infoToggle');
    const infoContent = document.getElementById('infoContent');
    
    function toggleInfoPanel() {
        if (infoContent && infoToggle) {
            if (infoContent.style.display === 'none' || !infoContent.style.display) {
                infoContent.style.display = 'block';
                infoToggle.textContent = '▲';
            } else {
                infoContent.style.display = 'none';
                infoToggle.textContent = '▼';
            }
        }
    }
    
    if (infoPanelHeader && infoContent && infoToggle) {
        // Клик на весь заголовок открывает/закрывает панель
        infoPanelHeader.addEventListener('click', function(e) {
            // Предотвращаем двойной вызов, если клик был на саму кнопку
            if (e.target === infoToggle) {
                toggleInfoPanel();
            } else {
                toggleInfoPanel();
            }
        });
        
        // Также обрабатываем клик на кнопку отдельно (на случай если нужно)
        infoToggle.addEventListener('click', function(e) {
            e.stopPropagation(); // Останавливаем всплытие, чтобы не было двойного вызова
            toggleInfoPanel();
        });
    }

    // Кнопка очистки результатов
    const clearResultsBtn = document.getElementById('clearResults');
    if (clearResultsBtn) {
        clearResultsBtn.addEventListener('click', function() {
            const resultDiv = document.getElementById('result');
            if (resultDiv) {
                resultDiv.innerHTML = '';
            }
            clearResultsBtn.style.display = 'none';
            // Очищаем превью и форму
            const fileInput = document.getElementById('file');
            if (fileInput) {
                fileInput.value = '';
            }
            const fileInfo = document.getElementById('fileInfo');
            if (fileInfo) {
                fileInfo.textContent = 'Файл не выбран';
            }
            hidePreview();
        });
    }
});
// --- КОНЕЦ НОВОЙ ЛОГИКИ ---