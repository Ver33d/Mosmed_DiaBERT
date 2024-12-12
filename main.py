import os
import pydicom  # Для проверки DICOM
import PyPDF2  # Для проверки PDF
import torch
import numpy as np
import json
import cv2
import requests
from fastapi.responses import JSONResponse
from typing import List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydicom import FileDataset
from PIL import Image

cv2.setNumThreads(0)

app = FastAPI()

# Подключаем шаблоны Jinja2
#templates = Jinja2Templates(directory="templates")

base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "../templates"))
templates_dir = os.path.join(base_dir, "../templates")


# UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
#
# static = os.path.join(os.getcwd(), "static")

base_dir = os.path.dirname(os.path.abspath(__file__))

# Строим пути относительно base_dir (без начального /)
static = os.path.join(base_dir, "../static")
UPLOAD_DIR = os.path.join(base_dir, "../uploads")
UPLOAD_FOLDER = os.path.join(base_dir, "../uploaded_files")


print("base_dir :",static)
print("base_dir :",UPLOAD_DIR)


if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Подключаем статические файлы (CSS и JS)
app.mount("/static", StaticFiles(directory=static), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Путь для хранения загруженных файлов
# UPLOAD_FOLDER = 'uploaded_files'
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Проверка на правильность формата DICOM
def is_valid_dicom(file_path: str) -> bool:
    try:
        pydicom.dcmread(file_path)
        return True
    except Exception:
        return False


# Проверка на правильность формата PDF
def is_valid_pdf(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages) > 0
    except Exception:
        return False


class ImageProcessor:
    image_size = 518, 518
    image_mean = 0.5
    image_std = 0.2865

    def __call__(self, image: Image):  # -> torch.tensor()
        """Переводит изображение из формата PIL.Image в формат torch.Tensor.

        Args:
            image (Image): Изображение в формате PIL (grayscale).

        Returns:
            torch.Tensor: Тензор размера (1, 3, H, W).
        """
        # PIL -> numpy
        image = np.array(image)
        print(torch.__version__)
        # resize
        if image.shape != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

        # to (0, 1) range
        image = image.astype('float32') / 255

        # normalization
        image = (image - self.image_mean) / self.image_std

        # grayscale -> to rgb
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # HWC -> CHW
        image = np.moveaxis(image, -1, 0)

        # numpy -> torch
        image = torch.from_numpy(image)

        # add batch dim
        image = image.unsqueeze(0)

        return image

    @staticmethod
    def preprocess_dicom(dicom: FileDataset) -> Image:
        """Извлекает изображение из DICOM объекта, применяет гистограммную эквализацию,
        изменяет размер изображения и преобразует в формат PIL.

        Args:
            dicom (FileDataset):
                Объект pydicom.FileDataset, обычно получаемый с помощью функции pydicom.dcmread.

        Returns:
            Image:
                Предобработанное изображение в формате PIL.
        """
        # dicom -> numpy
        image: np.ndarray = dicom.pixel_array

        # инвертируем пиксельные значения, если нужно
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            image = image.max() - image

        # переводим в uint8
        a, b = image.min(), image.max()
        if a == b:
            raise ValueError('Входное изображение содержит только одинаковые значения (image.min() == image.max())')
        image = ((image - a) / (b - a) * 255).astype('uint8')

        # эквализация гистограммы
        image = cv2.equalizeHist(image)

        # изменение размера
        image = cv2.resize(image, ImageProcessor.image_size, interpolation=cv2.INTER_AREA)

        # numpy -> PIL
        image = Image.fromarray(image)

        return image


# Главная страница с формой загрузки
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Запрос на главную страницу")
    return templates.TemplateResponse("index.html", {"request": request})


# Эндпоинт для загрузки файла
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Сохраняем файл на сервере
    with open(file_path, "wb") as f:
        print("file_path 1 ", file_path)
        f.write(await file.read())

    # Обработка форматов
    if file.filename.lower().endswith('.dcm'):
        if not is_valid_dicom(file_path):
            os.remove(file_path)  # Удаляем неверный файл
            raise HTTPException(status_code=400, detail="Ошибка: Неверный формат файла. Требуется DICOM.")

        # Обрабатываем DICOM файл с использованием preprocess_dicom
        dicom = pydicom.dcmread(file_path)
        try:
            pil_image = ImageProcessor.preprocess_dicom(dicom)
        except ValueError as e:
            os.remove(file_path)  # Удаляем неверный файл
            raise HTTPException(status_code=400, detail=f"Ошибка при обработке DICOM: {str(e)}")

        # Сохраняем изображение как PNG в папку uploads
        png_filename = os.path.join("../uploads", f"{file.filename.split('.')[0]}.png")
        pil_image.save(os.path.join(UPLOAD_DIR, f"{file.filename.split('.')[0]}.png"), format="PNG")

        # Преобразуем изображение в тензор
        image_processor = ImageProcessor()
        tensor_image = image_processor(pil_image)

        png_file_path = png_filename
        print(" png_filename", png_filename)

    elif file.filename.lower().endswith('.pdf'):
        if not is_valid_pdf(file_path):
            os.remove(file_path)  # Удаляем неверный файл
            raise HTTPException(status_code=400, detail="Ошибка: Неверный формат файла. Требуется PDF.")
    else:
        os.remove(file_path)  # Удаляем файл с неподдерживаемым форматом
        raise HTTPException(status_code=400, detail="Ошибка: Неподдерживаемый формат файла.")

    return {"message": f"Файл успешно загружен!", "image_path": png_filename}  # {file.filename}


@app.post("/process/")
async def process_image(request: Request, file: UploadFile = File(...), options: List[str] = Form(...)):
    uploaded_file_path = '../uploads/1.png'
    uploaded_file_path_2 = os.path.join(UPLOAD_FOLDER, file.filename)
    options = options[0]
    options = json.loads(options)

    # Собираем информацию, которую нужно будет показать на странице
    result = {}
    if options[0] == "on":
        url = "https://a7f1-193-41-143-66.ngrok-free.app/predict_proba_clav_fracture/"
        files = {'file': open(uploaded_file_path_2, 'rb')}
        response = requests.post(url, files=files)
        response_json = response.json()

        # Извлекаем вероятность из ответа
        probability = response_json.get('probability', 0)
        # Формируем строку и записываем в переменную
        result['clavicle_fracture'] = f"Перелом ключицы обнаружен - вероятность {probability:.2f}"

    if options[1] == "on":
        url = "https://a7f1-193-41-143-66.ngrok-free.app/predict_proba_medimp/"
        files = {'file': open(uploaded_file_path_2, 'rb')}
        response = requests.post(url, files=files)
        response_json = response.json()

        # Извлекаем вероятность из ответа
        probability = response_json.get('probability', 0)

        # Формируем строку и записываем в переменную
        result['clavicle_fracture'] = f"Наличие посторонних предметов обнаружено - вероятность {probability:.2f}"

    if options[2] == "on":
        url = "https://a7f1-193-41-143-66.ngrok-free.app/predict_segmentation_clav_fracture/"
        files = {'file': open(uploaded_file_path_2, 'rb')}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            # Генерируем имя для сохранённого файла (можно использовать уникальное имя)
            file_name = 'clavicle_segmentation.png'

            # Сохраняем изображение
            saved_image_path = save_image_from_response(response, file_name)

            # Указываем путь к сохранённому изображению в result
            result['clavicle_segmentation'] = saved_image_path
        else:
            print(f"Ошибка при получении изображения: {response.status_code}")

    if options[3] == "on":
        url = "https://a7f1-193-41-143-66.ngrok-free.app/predict_proba_medimp/"
        files = {'file': open(uploaded_file_path_2, 'rb')}
        response = requests.post(url, files=files)
        response_json = response.json()

        # Извлекаем вероятность из ответа
        probability = response_json.get('probability', 0)
        if (probability > 0.5):
            url = "https://a7f1-193-41-143-66.ngrok-free.app/predict_segmentation_medimp/"
            files = {'file': open(uploaded_file_path_2, 'rb')}
            response = requests.post(url, files=files)
            # Проверяем, что запрос прошёл успешно
            if response.status_code == 200:
                # Генерируем имя для сохранённого файла (можно использовать уникальное имя)
                file_name = 'clavicle_segmentation.png'

                # Сохраняем изображение
                saved_image_path = save_image_from_response(response, file_name)

                # Указываем путь к сохранённому изображению в result
                result['foreign_objects_segmentation'] = saved_image_path  # Путь к изображению

            else:
                print(f"Ошибка при получении изображения: {response.status_code}")
        else:
            pass
    if options[4] == "on":
        result['foreign_objects_description'] = 'Найден предмет, здесь будет подробное описание от LLM. '

    # if options[5] == "on":
    #     print("Генерация отчета")
    #     result['generate_report'] = True  # Кнопка для генерации отчета
    # Возвращаем ответ в формате JSON
    return JSONResponse(content={"message": "Изображение успешно обработано", "result": result})


@app.get("/processed_image", response_class=HTMLResponse)
async def processed_image(request: Request):
    return templates.TemplateResponse("process.html", {"request": request})


# Функция для сохранения картинки
def save_image_from_response(response, filename):
    image_data = response.content  # Получаем байты изображения

    file_path =  f"/uploads/{filename}"
    with open(file_path, 'wb') as f:
        f.write(image_data)  # Записываем байты в файл
    return file_path  # Возвращаем путь к сохранённому файлу


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
