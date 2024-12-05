from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pydicom  # Для проверки DICOM
import PyPDF2  # Для проверки PDF

app = FastAPI()

# Подключаем шаблоны Jinja2
templates = Jinja2Templates(directory="templates")  # Укажите папку с вашими HTML шаблонами

# Настройка папок для хранения файлов
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Подключаем статические файлы (CSS и JS)
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Путь для хранения загруженных файлов
UPLOAD_FOLDER = 'uploaded_files'


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


# Главная страница с формой загрузки
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Эндпоинт для загрузки файла
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Сохраняем файл на сервере
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Определение формата файла
    if file.filename.lower().endswith('.dcm'):
        if not is_valid_dicom(file_path):
            os.remove(file_path)  # Удаляем неверный файл
            raise HTTPException(status_code=400, detail="Ошибка: Неверный формат файла. Требуется DICOM.")
    elif file.filename.lower().endswith('.pdf'):
        if not is_valid_pdf(file_path):
            os.remove(file_path)  # Удаляем неверный файл
            raise HTTPException(status_code=400, detail="Ошибка: Неверный формат файла. Требуется PDF.")
    else:
        os.remove(file_path)  # Удаляем файл с неподдерживаемым форматом
        raise HTTPException(status_code=400, detail="Ошибка: Неподдерживаемый формат файла.")

    return {"message": f"Файл {file.filename} успешно загружен!"}
