from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import os
import tempfile
import uuid
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Использовать бэкенд без GUI для matplotlib
import logging
from pydantic import BaseModel
from typing import List, Optional
import httpx  # Для загрузки файлов по URL
from urllib.parse import urlparse
import mimetypes
from models.model import ImageClassifier

# --- НАСТРОЙКА ПРИЛОЖЕНИЯ ---
app = FastAPI(title="Deepfake Detector API")
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ЗАГРУЗКА МОДЕЛИ ---
MODEL_PATH = "models/best_model.pth"
MODEL_NAME = 'vit_base_patch16_clip_224.openai'

# Создаем экземпляр модели напрямую
model = ImageClassifier(model_name=MODEL_NAME)

# Загружаем state_dict
try:
    state_dict = torch.load(MODEL_PATH, map_location='cpu') # Загружаем на CPU, потом переместим
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"ImageClassifier модель загружена на {device}")
except Exception as e:
    logger.error(f"Ошибка загрузки ImageClassifier модели: {e}")
    raise e

# --- ДЕТЕКТОР ЛИЦ ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- АУГМЕНТАЦИИ (только валидационные) ---
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2()
])

# --- ПАРАМЕТРЫ ---
MIN_FACE_SIZE_PX = 220  # Минимальный размер (ширина или высота) кропа лица в пикселях
DEFAULT_FRAMES_TO_SAVE = 15  # Количество кадров по умолчанию
MAX_FRAMES_TO_SAVE = 25  # Максимальное количество кадров, которое можно запросить
MIN_FRAMES_TO_SAVE = 5  # Минимальное количество кадров
DEFAULT_THRESHOLD = 0.5  # Порог уверенности по умолчанию (вердикт: >0.5 = оригинал)
MIN_THRESHOLD = 0.3  # Минимальный порог уверенности
MAX_THRESHOLD = 0.7  # Максимальный порог уверенности
# ---

# --- Pydantic Модели ---
class FaceResult(BaseModel):
    face_index: int
    bbox: dict
    prediction: str
    probability: float
    heatmap: Optional[str] = None
    face_crop_image: Optional[str] = None

class DetectedFaceResult(BaseModel):
    filename: str
    verdict: str
    probability: float
    frame_index: int
# ---

class ImageResult(BaseModel):
    annotated_image: str
    face_results: List[FaceResult]

class VideoResult(BaseModel):
    summary: str
    annotated_video: str
    plot: Optional[str] = None
    detected_faces: List[DetectedFaceResult]

class UploadResponse(BaseModel):
    message: str
    result: Optional[ImageResult] = None
    result_video: Optional[VideoResult] = None

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def detect_faces(image):
    """Обнаруживает лица на изображении с помощью MediaPipe.
    
    Args:
        image: Изображение в формате BGR (OpenCV)
        
    Returns:
        Список словарей с координатами bbox для каждого обнаруженного лица
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    faces = []
    if results.detections:
        h, w, _ = image.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox = {
                'xmin': int(bboxC.xmin * w),
                'ymin': int(bboxC.ymin * h),
                'width': int(bboxC.width * w),
                'height': int(bboxC.height * h)
            }

            bbox['xmax'] = bbox['xmin'] + bbox['width']
            bbox['ymax'] = bbox['ymin'] + bbox['height']
            # Добавляем паддинг 50% для лучшего кропа лица
            padding_x = int(bbox['width'] * 0.5)
            padding_y = int(bbox['height'] * 0.5)

            # Рассчитываем новые координаты с отступом
            new_xmin = bbox['xmin'] - padding_x
            new_ymin = bbox['ymin'] - padding_y
            new_xmax = bbox['xmax'] + padding_x
            new_ymax = bbox['ymax'] + padding_y

            # Ограничиваем координаты размерами изображения
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(w, new_xmax)
            new_ymax = min(h, new_ymax)

            # Обновляем bbox (это bbox с паддингом)
            bbox_with_padding = {
                'xmin': new_xmin,
                'ymin': new_ymin,
                'xmax': new_xmax,
                'ymax': new_ymax,
                'width': new_xmax - new_xmin,
                'height': new_ymax - new_ymin
            }

            # --- ПРОВЕРКА МИНИМАЛЬНОГО РАЗМЕРА ---
            if bbox_with_padding['width'] >= MIN_FACE_SIZE_PX and bbox_with_padding['height'] >= MIN_FACE_SIZE_PX:
                faces.append(bbox_with_padding) # Добавляем bbox с паддингом, если он достаточно большой
            # else:
            #     print(f"Обнаруженное лицо игнорируется из-за маленького размера: {bbox_with_padding['width']}x{bbox_with_padding['height']} < {MIN_FACE_SIZE_PX}px")
            # ---
    return faces

def preprocess_face(face_image):
    """Применяет аугментации к вырезанному лицу для подачи в модель.
    
    Args:
        face_image: Кроп лица в формате BGR
        
    Returns:
        Тензор изображения (C, H, W), нормализованный и приведенный к 224x224
    """
    transformed = val_transforms(image=face_image)
    return transformed['image']

# --- Grad-CAM для ViT (для изображений) ---
class ViTGradCam:
    def __init__(self, vit_model, target_layer):
        self.model = vit_model.eval()
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output shape: (B, N, C) для ViT блоков до head, удалим CLS токен позже
            self.feature = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape такой же как feature
            self.gradient = grad_output[0]

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    @staticmethod
    def _reshape_tokens_to_map(tensor, height=14, width=14):
        # tensor: (B, N, C), где N = 1 + H*W (включая CLS)
        tensor = tensor[:, 1:, :]  # убираем CLS
        b, n, c = tensor.shape
        assert n == height * width, f"Ожидалось {height*width} токенов, получено {n}"
        tensor = tensor.reshape(b, height, width, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        return tensor

    def __call__(self, input_tensor):
        # input_tensor: (1, 3, 224, 224), должен иметь requires_grad=True
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)  # logits (1,)
        score = output.squeeze()  # скаляр логит
        score.backward(retain_graph=True)

        # Получаем градиенты и признаки
        grad = self.gradient.detach()  # (B, N, C)
        feat = self.feature.detach()   # (B, N, C)

        # Преобразуем в карты (B, C, H, W)
        grad_map = self._reshape_tokens_to_map(grad)
        feat_map = self._reshape_tokens_to_map(feat)

        # Веса как средние по пространству
        weights = grad_map.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * feat_map).sum(dim=1, keepdim=False)  # (B, H, W)
        cam = torch.relu(cam)

        # Нормализация в [0,1]
        cam_min = cam.min(dim=-1, keepdim=False)[0].min(dim=-2, keepdim=False)[0].unsqueeze(-1).unsqueeze(-1)
        cam_max = cam.max(dim=-1, keepdim=False)[0].max(dim=-2, keepdim=False)[0].unsqueeze(-1).unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_np = cam[0].cpu().numpy()  # (H, W)
        cam_np = cv2.resize(cam_np, (224, 224))
        return cam_np

    def close(self):
        for h in self.handlers:
            h.remove()

def overlay_heatmap_on_image(rgb_image_224, heatmap_224, alpha=0.5):
    # rgb_image_224: np.uint8 (H,W,3) в RGB; heatmap_224: float32 [0,1]
    heat = np.uint8(255 * heatmap_224)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1 - alpha) * rgb_image_224.astype(np.float32) + alpha * heat.astype(np.float32), 0, 255)
    return overlay.astype(np.uint8)
# --- конец Grad-CAM ---

def predict_image(image_path, threshold: float = DEFAULT_THRESHOLD):
    """Предсказывает дипфейк для одного изображения.
    
    Args:
        image_path: Путь к изображению
        threshold: Порог уверенности (по умолчанию 0.5)
        
    Returns:
        ImageResult с аннотированным изображением и результатами для каждого лица
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    faces = detect_faces(image)
    if not faces:
        raise ValueError("Лица не обнаружены на изображении (или все обнаруженные лица были отфильтрованы по минимальному размеру).")

    results = []
    for i, face_bbox in enumerate(faces):
        face_crop = image[face_bbox['ymin']:face_bbox['ymax'], face_bbox['xmin']:face_bbox['xmax']]

        if face_crop.size == 0:
            results.append({"face_index": i, "error": f"Обнаруженное лицо {i} пустое или неверно вырезано (bbox: {face_bbox})."})
            continue

        # Препроцессинг
        face_tensor = preprocess_face(face_crop)  # (C, H, W)
        face_tensor = face_tensor.unsqueeze(0)  # (1, C, H, W) - Подаем одно изображение
        face_tensor = face_tensor.to(device)

        with torch.no_grad():
            logits = model(face_tensor)  # (1,)
            prob = torch.sigmoid(logits).cpu().item()

        # Определяем вердикт на основе порога
        verdict = "Оригинал" if prob > threshold else "Дипфейк"

        # Сохранение кропа лица с размером 224x224
        face_crop_resized = cv2.resize(face_crop, (224, 224))
        face_crop_filename = f"face_result_{i}_{os.path.basename(image_path)}"
        face_crop_path = os.path.join(UPLOAD_DIR, face_crop_filename)
        cv2.imwrite(face_crop_path, face_crop_resized)
        # ---

        # --- ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ (Grad-CAM для ViT) ---
        heatmap_filename = None
        try:
            # Подготовим вход для градиентов
            face_tensor_gc = face_tensor.clone().detach().requires_grad_(True)
            target_layer = model.backbone.blocks[-1].norm1
            gc = ViTGradCam(model.backbone, target_layer)
            cam_map = gc(face_tensor_gc)  # (224,224) в [0,1]
            gc.close()

            # Готовим 224x224 RGB картинку из кропа для наложения
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_rgb_224 = cv2.resize(face_rgb, (224, 224))
            overlay = overlay_heatmap_on_image(face_rgb_224, cam_map, alpha=0.5)
            # Сохраняем
            heatmap_filename = f"heatmap_face_{i}_{os.path.basename(image_path)}"
            heatmap_path = os.path.join(UPLOAD_DIR, heatmap_filename)
            cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        except Exception as _:
            heatmap_filename = None
        # ---

        results.append({
            "face_index": i,
            "bbox": face_bbox,
            "prediction": verdict,
            "probability": prob,
            "heatmap": heatmap_filename,
            "face_crop_image": face_crop_filename
        })

    # Рисуем bbox на оригинальном изображении
    annotated_image_filename = f"annotated_{os.path.basename(image_path)}"
    annotated_image_path = os.path.join(UPLOAD_DIR, annotated_image_filename)
    annotated_image = image.copy()
    img_h, img_w = image.shape[:2]
    for res in results:
        if "error" not in res:
            bbox = res["bbox"]
            prob = res["probability"]
            # Пропускаем отрисовку, если bbox покрывает почти всё изображение
            bbox_area = max(0, bbox['width']) * max(0, bbox['height'])
            img_area = max(1, img_w * img_h)
            if bbox_area / img_area < 0.9:
                cv2.rectangle(annotated_image, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']),
                              (0, 255, 0), 2)

    cv2.imwrite(annotated_image_path, annotated_image)

    return ImageResult(annotated_image=annotated_image_filename, face_results=results)


def create_timeline_plot(timeline_data, fps, video_path, threshold: float = DEFAULT_THRESHOLD):
    """Создает интерактивный график вероятности по времени с визуализацией дипфейков.
    
    Args:
        timeline_data: Список кортежей (время_сек, вероятность, есть_дипфейк)
        fps: Частота кадров видео
        video_path: Путь к видео файлу
        threshold: Порог уверенности для определения дипфейка
        
    Returns:
        Имя файла сохраненного графика или None
    """
    if not timeline_data:
        return None
    
    # Извлекаем данные
    times = [t[0] for t in timeline_data]
    probs = [t[1] for t in timeline_data]
    has_deepfake = [t[2] for t in timeline_data]
    
    # Настройка стиля matplotlib для темной темы
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#1a1a1a')
    ax.set_facecolor('#2d2d2d')
    
    # Определяем цвета для точек и областей на основе порога
    colors = ['#0be881' if not df else '#ff4e50' for df in has_deepfake]
    
    # Рисуем область под графиком с градиентом (оригинал/дипфейк)
    ax.fill_between(times, probs, threshold, where=[p >= threshold for p in probs], 
                     color='#0be881', alpha=0.2, label='Оригинал')
    ax.fill_between(times, probs, threshold, where=[p < threshold for p in probs], 
                     color='#ff4e50', alpha=0.2, label='Дипфейк')
    
    # Рисуем линию вероятности
    ax.plot(times, probs, color='#ffffff', linewidth=2, alpha=0.6, zorder=1)
    
    # Рисуем точки на графике с цветами
    for i, (t, p, df) in enumerate(timeline_data):
        ax.scatter(t, p, color=colors[i], s=80, zorder=3, 
                  edgecolors='white', linewidths=1.5, alpha=0.9)
    
    # Пороговая линия (динамическая)
    ax.axhline(y=threshold, color='#ffa801', linestyle='--', linewidth=2, 
              alpha=0.7, label=f'Порог ({threshold:.2f})', zorder=2)
    
    # Настройка осей
    ax.set_xlabel('Время (секунды)', fontsize=12, color='#f0f0f0', fontweight='bold')
    ax.set_ylabel('Вероятность оригинальности', fontsize=12, color='#f0f0f0', fontweight='bold')
    ax.set_title('Временная шкала вероятности: Обнаружение дипфейков', 
                 fontsize=14, color='#ffffff', fontweight='bold', pad=20)
    
    # Настройка сетки
    ax.grid(True, alpha=0.2, color='#ffffff', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Настройка границ
    ax.set_ylim([0, 1])
    if times:
        ax.set_xlim([0, max(times) * 1.05])
    
    # Форматирование меток осей
    ax.tick_params(colors='#a0a0a0', labelsize=10)
    
    # Легенда
    legend = ax.legend(loc='upper right', framealpha=0.9, 
                       facecolor='#3a3a3a', edgecolor='#444', 
                       fontsize=10, labelcolor='#f0f0f0')
    legend.get_frame().set_linewidth(1.5)
    
    # Добавляем маркеры для дипфейков на временной шкале
    deepfake_times = [t[0] for t in timeline_data if t[2]]
    if deepfake_times:
        # Выделяем сегменты с дипфейками вертикальными линиями
        for df_time in deepfake_times:
            ax.axvline(x=df_time, color='#ff4e50', linestyle=':', 
                      linewidth=1.5, alpha=0.5, zorder=0)
    
    # Добавляем статистику
    total_deepfake_frames = sum(has_deepfake)
    total_frames = len(has_deepfake)
    deepfake_percentage = (total_deepfake_frames / total_frames * 100) if total_frames > 0 else 0
    
    stats_text = f'Дипфейк обнаружен: {total_deepfake_frames}/{total_frames} кадров ({deepfake_percentage:.1f}%)'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, color='#ff4e50' if total_deepfake_frames > 0 else '#0be881',
            bbox=dict(boxstyle='round', facecolor='#3a3a3a', alpha=0.8, edgecolor='#444', linewidth=1.5),
            verticalalignment='bottom', fontweight='bold')
    
    # Сохраняем график
    plot_filename = f"timeline_plot_{os.path.basename(video_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(UPLOAD_DIR, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor='#1a1a1a', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    return plot_filename


def predict_video(video_path, requested_frames_to_save: int = DEFAULT_FRAMES_TO_SAVE, 
                 frame_skip: int = 5, threshold: float = DEFAULT_THRESHOLD):
    """Предсказывает дипфейк для видео.
    
    Args:
        video_path: Путь к видео файлу
        requested_frames_to_save: Количество кадров для сохранения (5-25)
        frame_skip: Пропуск кадров (1 = все кадры, 5 = каждый 5-й)
        threshold: Порог уверенности для определения дипфейка
        
    Returns:
        VideoResult с аннотированным видео, графиком и обнаруженными лицами
    """
    cap = cv2.VideoCapture(video_path)
    frame_probs = [] # Теперь это будет список списков: [[prob_face1, prob_face2, ...], [prob_face1, prob_face2, ...], ...] для кадров
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # Значение по умолчанию, если не удалось получить
    annotated_frames = []

    # --- ХРАНЕНИЕ ЛИЦ ---
    all_valid_faces_data = [] # Список словарей {'face_crop': ..., 'frame_index': ..., 'verdict': ..., 'probability': ...}
    # ---

    # Используем переданный frame_skip (1 = все кадры, 5 = каждый 5-й)

    processed_frame_index = 0  # Счетчик обработанных кадров (для расчета времени)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Пропуск кадров согласно режиму обработки
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        faces = detect_faces(frame) # Теперь detect_faces может отфильтровать маленькие лица
        current_frame_probs = [] # Вероятности для всех лиц в текущем кадре

        for face_bbox in faces: # Обрабатываем *все* подходящие лица в кадре
            face_crop = frame[face_bbox['ymin']:face_bbox['ymax'], face_bbox['xmin']:face_bbox['xmax']]

            if face_crop.size != 0:
                face_tensor = preprocess_face(face_crop) # (C, H, W)
                face_tensor = face_tensor.unsqueeze(0) # (1, C, H, W) - Подаем одно изображение
                face_tensor = face_tensor.to(device)

                with torch.no_grad():
                    logits = model(face_tensor) # (1,)
                    frame_prob = torch.sigmoid(logits).cpu().item()
                current_frame_probs.append(frame_prob)

                # Сохраняем данные лица (кроп, индекс кадра, вердикт, вероятность) во временный список
                verdict = "Оригинал" if frame_prob > threshold else "Дипфейк"
                all_valid_faces_data.append({
                    'face_crop': face_crop,
                    'frame_index': frame_count,
                    'verdict': verdict,
                    'probability': frame_prob
                })
                # ---

        # --- ОБНОВЛЕНИЕ: Добавляем вероятности для всех лиц в кадре (или None, если лиц не было) ---
        if current_frame_probs:
            frame_probs.append(current_frame_probs) # Добавляем список вероятностей для этого кадра
        else:
            frame_probs.append(None) # Добавляем None для кадра без (допустимых) лиц, чтобы сохранить порядок
        # ---

        # Рисуем bbox и вероятность на кадре
        annotated_frame = frame.copy()
        if current_frame_probs:
            # Визуализируем все лица в кадре с учетом порога
            frame_h, frame_w = annotated_frame.shape[:2]
            for i, (face_bbox, prob) in enumerate(zip(faces, current_frame_probs)):
                verdict = "Оригинал" if prob > threshold else "Дипфейк"
                color = (0, 255, 0) if verdict.startswith("Оригинал") else (0, 0, 255) # Зелёный для оригинала, красный для дипфейка
                # Пропускаем отрисовку, если bbox покрывает почти весь кадр
                fb_area = max(0, face_bbox['width']) * max(0, face_bbox['height'])
                fr_area = max(1, frame_w * frame_h)
                if fb_area / fr_area < 0.9:
                    cv2.rectangle(annotated_frame, (face_bbox['xmin'], face_bbox['ymin']), (face_bbox['xmax'], face_bbox['ymax']), color, 2)
                    cv2.putText(annotated_frame, f'{verdict} (Face {i}, Prob: {prob:.2f})', (face_bbox['xmin'], face_bbox['ymin'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(annotated_frame, 'No Face (or too small)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Обновим текст, если лицо не найдено или слишком маленькое

        annotated_frames.append(annotated_frame)
        frame_count += 1
        processed_frame_index += 1  # Увеличиваем счетчик обработанных кадров

    cap.release()

    # --- ВЫПОЛНЯЕМ РАВНОМЕРНУЮ ВЫБОРКУ И СОХРАНЕНИЕ ---
    detected_faces = [] # Список для хранения обнаруженных лиц (теперь список DetectedFaceResult)
    total_valid_faces = len(all_valid_faces_data)
    if total_valid_faces > 0:
        # --- ИСПОЛЬЗУЕМ ЗАПРОШЕННОЕ КОЛИЧЕСТВО КАДРОВ ---
        # Ограничиваем запрашиваемое количество кадров максимальным/минимальным значением
        requested_frames_to_save = max(MIN_FRAMES_TO_SAVE, min(MAX_FRAMES_TO_SAVE, requested_frames_to_save))
        # ---
        if total_valid_faces <= requested_frames_to_save:
            # Если подходящих лиц меньше или равно requested_frames_to_save, сохраняем все
            selected_indices = list(range(total_valid_faces))
        else:
            # Вычисляем шаг для равномерной выборки
            step = total_valid_faces / requested_frames_to_save
            selected_indices = [int(i * step) for i in range(requested_frames_to_save)]

        for idx in selected_indices:
            face_data = all_valid_faces_data[idx]
            face_crop = face_data['face_crop']
            frame_idx = face_data['frame_index']
            verdict = face_data['verdict']
            prob = face_data['probability']

            # Генерируем уникальное имя файла для лица
            face_filename = f"face_frame_{frame_idx}_idx_{idx}_{uuid.uuid4().hex}.jpg"
            face_path = os.path.join(UPLOAD_DIR, face_filename)
            # Сохраняем лицо
            cv2.imwrite(face_path, face_crop)
            # Добавляем путь к лицу, verdict и prob в список как DetectedFaceResult
            detected_faces.append(DetectedFaceResult(
                filename=face_filename,
                verdict=verdict,
                probability=prob,
                frame_index=frame_idx # Сохраняем индекс кадра
            ))

    # --- ОСТАЛЬНАЯ ЧАСТЬ ФУНКЦИИ (создание видео, обработка frame_probs, возврат результата) ---
    # Создание временного файла для видео с аннотациями
    annotated_video_filename = f"annotated_{os.path.basename(video_path)}"
    annotated_video_path = os.path.join(UPLOAD_DIR, annotated_video_filename)
    if annotated_frames:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Попробуем XVID
        # ---
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (annotated_frames[0].shape[1], annotated_frames[0].shape[0]))
        for f in annotated_frames:
            out.write(f)
        out.release()

    # --- Учитываем пропущенные кадры при построении графика ---
    # Фильтруем None значения для вычисления средней
    # Теперь frame_probs - список списков. Нужно "сплющить" его, чтобы получить все вероятности
    all_valid_probs = []
    # Собираем данные для графика: время и вероятность для каждого кадра с лицами
    timeline_data = []  # Список кортежей (время_в_секундах, средняя_вероятность_для_кадра, есть_ли_дипфейк)
    
    timeline_index = 0  # Индекс для временной шкалы (с учетом пропусков)
    for probs_list in frame_probs:
        if probs_list is not None:
            all_valid_probs.extend(probs_list)
            # Вычисляем среднюю вероятность для этого кадра
            avg_prob_frame = sum(probs_list) / len(probs_list)
            # Вычисляем время в секундах с учетом frame_skip
            time_seconds = (timeline_index * frame_skip) / fps
            # Определяем есть ли дипфейк в кадре (если хотя бы одно лицо < threshold)
            has_deepfake = any(p < threshold for p in probs_list)
            timeline_data.append((time_seconds, avg_prob_frame, has_deepfake))
        timeline_index += 1

    # Генерация графика временной шкалы с учетом порога
    plot_filename = None
    if timeline_data:
        plot_filename = create_timeline_plot(timeline_data, fps, video_path, threshold)

    # Агрегированная вероятность только по кадрам с лицами
    if all_valid_probs:
        avg_prob = sum(all_valid_probs) / len(all_valid_probs)
        verdict_for_summary = "Оригинал" if avg_prob > threshold else "Дипфейк"
        summary_text = f"Видео: {verdict_for_summary} (Средняя вероятность реальности: {avg_prob:.4f})"
        summary_text += f" (Обработано {len(all_valid_probs)} лиц из ~{len(frame_probs)} кадров)."
    else:
        summary_text = f"Лица не обнаружены (или все слишком малы) ни в одном из {len(frame_probs)} обработанных кадров."

    # --- ВОЗВРАЩАЕМ ПУТИ К СОХРАНЁННЫМ ЛИЦАМ ---
    return VideoResult(summary=summary_text, annotated_video=annotated_video_filename, plot=plot_filename, detected_faces=detected_faces)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...), 
    file_type: str = Form(...), 
    frames_count: Optional[int] = Form(DEFAULT_FRAMES_TO_SAVE),
    threshold: Optional[float] = Form(DEFAULT_THRESHOLD),
    speed_mode: Optional[str] = Form("fast")
):
    # Проверка типа файла
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "video/mp4", "video/avi", "video/mov", "video/quicktime"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла. Пожалуйста, загрузите изображение (JPEG, PNG) или видео (MP4, AVI, MOV).")

    # Проверка размера файла (100 МБ)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МБ в байтах
    
    # Генерация уникального имени файла
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, unique_filename)

    # Сохранение файла с проверкой размера
    file_size = 0
    with open(filepath, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # Читаем по 1 МБ
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                # Удаляем файл, если он превышает лимит
                buffer.close()
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise HTTPException(status_code=400, detail=f"Файл слишком большой. Максимальный размер: 100 МБ. Ваш файл: {file_size / (1024 * 1024):.2f} МБ")
            buffer.write(chunk)

    # Валидация и нормализация параметров
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
    
    try:
        if file_type == 'image':
            result = predict_image(filepath, threshold=threshold)
            return UploadResponse(message="Изображение успешно обработано.", result=result)

        elif file_type == 'video':
            # Валидация frames_count
            if frames_count is None:
                frames_count = DEFAULT_FRAMES_TO_SAVE
            frames_count = max(MIN_FRAMES_TO_SAVE, min(MAX_FRAMES_TO_SAVE, frames_count))
            
            # Определяем frame_skip в зависимости от режима скорости
            frame_skip = 1 if speed_mode == "precise" else 5
            
            result = predict_video(filepath, requested_frames_to_save=frames_count, 
                                 frame_skip=frame_skip, threshold=threshold)
            return UploadResponse(message="Видео успешно обработано.", result_video=result)

        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Must be 'image' or 'video'.")

    except ValueError as ve:
        # Удалить файл при ошибке
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {filepath}: {e}")
        # Удалить файл при ошибке
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


# --- ФУНКЦИЯ ЗАГРУЗКИ ФАЙЛА ПО URL ---
async def download_file_from_url(url: str, max_size: int = 100 * 1024 * 1024) -> tuple[str, str]:
    """
    Загружает файл по URL и сохраняет его локально.
    
    Args:
        url: URL файла для загрузки
        max_size: Максимальный размер файла в байтах (по умолчанию 100 МБ)
    
    Returns:
        tuple: (путь_к_файлу, content_type)
    
    Raises:
        HTTPException: Если URL невалидный, файл слишком большой или произошла ошибка загрузки
    """
    # Валидация URL
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Некорректный URL. Пожалуйста, укажите полный URL (например: https://example.com/image.jpg)")
    
    # Определяем расширение файла из URL
    file_ext = os.path.splitext(parsed.path)[1]
    if not file_ext:
        # Пытаемся определить по content-type
        file_ext = '.jpg'  # По умолчанию
    
    # Генерируем уникальное имя файла
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    filepath = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        # Загружаем файл по URL
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async with client.stream('GET', url) as response:
                # Проверяем статус код
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Не удалось загрузить файл по URL. Статус: {response.status_code}")
                
                # Получаем content-type
                content_type = response.headers.get('content-type', '')
                
                # Проверяем тип контента
                allowed_types = {"image/jpeg", "image/jpg", "image/png", "video/mp4", "video/avi", "video/mov", "video/quicktime", "image/webp"}
                if content_type and not any(allowed_type in content_type.lower() for allowed_type in allowed_types):
                    # Если content-type не подходит, проверяем по расширению
                    ext_lower = file_ext.lower()
                    if ext_lower not in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webp']:
                        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла. Поддерживаются: JPEG, PNG, MP4, AVI, MOV")
                
                # Сохраняем файл с проверкой размера
                file_size = 0
                with open(filepath, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # Читаем по 1 МБ
                        file_size += len(chunk)
                        if file_size > max_size:
                            f.close()
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            raise HTTPException(status_code=400, 
                                             detail=f"Файл слишком большой. Максимальный размер: 100 МБ. Размер файла: {file_size / (1024 * 1024):.2f} МБ")
                        f.write(chunk)
        
        logger.info(f"Файл успешно загружен по URL: {url} -> {filepath}")
        return filepath, content_type
        
    except httpx.TimeoutException:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=408, detail="Превышено время ожидания при загрузке файла. Попробуйте другой URL или проверьте подключение к интернету.")
    except httpx.RequestError as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Ошибка при загрузке файла: {str(e)}")
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Неожиданная ошибка при загрузке файла по URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")
# ---


# --- ENDPOINT ДЛЯ ЗАГРУЗКИ ПО URL ---
@app.post("/upload-url", response_model=UploadResponse)
async def upload_file_from_url(
    url: str = Form(...), 
    file_type: str = Form(...), 
    frames_count: Optional[int] = Form(DEFAULT_FRAMES_TO_SAVE),
    threshold: Optional[float] = Form(DEFAULT_THRESHOLD),
    speed_mode: Optional[str] = Form("fast")
):
    """
    Загружает файл по URL и обрабатывает его.
    
    Args:
        url: URL изображения или видео
        file_type: Тип файла ('image' или 'video')
        frames_count: Количество кадров для видео (опционально)
    
    Returns:
        UploadResponse: Результат обработки файла
    """
    # Загружаем файл по URL
    try:
        filepath, content_type = await download_file_from_url(url)
    except HTTPException:
        raise  # Пробрасываем HTTPException как есть
    
    # Валидация и нормализация параметров
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))
    
    # Обрабатываем файл
    try:
        if file_type == 'image':
            result = predict_image(filepath, threshold=threshold)
            return UploadResponse(message="Изображение успешно обработано.", result=result)
        
        elif file_type == 'video':
            if frames_count is None:
                frames_count = DEFAULT_FRAMES_TO_SAVE
            frames_count = max(MIN_FRAMES_TO_SAVE, min(MAX_FRAMES_TO_SAVE, frames_count))
            
            # Определяем frame_skip в зависимости от режима скорости
            frame_skip = 1 if speed_mode == "precise" else 5
            
            result = predict_video(filepath, requested_frames_to_save=frames_count, 
                                 frame_skip=frame_skip, threshold=threshold)
            return UploadResponse(message="Видео успешно обработано.", result_video=result)
        
        else:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(status_code=400, detail="Invalid file type. Must be 'image' or 'video'.")
    
    except ValueError as ve:
        # Удалить файл при ошибке
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {filepath}: {e}")
        # Удалить файл при ошибке
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
# ---


# Маршрут для отдачи статических файлов (изображений, видео, графиков)
@app.get("/uploads/{filename}")
async def get_upload_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
