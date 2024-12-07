 $(document).ready(function() {
    const uploadArea = $('#upload-area');
    const fileInput = $('#file-input');
    const statusDiv = $('#upload-status');
    const processButton = $('#process-image'); // Кнопка для обработки
    const imageContainer = $('#image-container'); // Контейнер для изображения
    const checkboxContainer = $('#checkbox-container'); // Контейнер с флажками

    let uploadedFile; // Переменная для хранения загруженного файла

    // Drag-and-drop
    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        uploadArea.css('background-color', '#e0e0e0');
    });

    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        uploadArea.css('background-color', '#ffffff');
    });

    uploadArea.on('drop', function(e) {
        e.preventDefault();
        uploadArea.css('background-color', '#ffffff');
        const file = e.originalEvent.dataTransfer.files[0];
        handleFileUpload(file);
    });

    // File input change
    fileInput.on('change', function(e) {
        const file = e.target.files[0];
        handleFileUpload(file);
    });

    function handleFileUpload(file) {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        $.ajax({
            url: '/upload/', // Проверьте этот URL на сервере
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                uploadedFile = file; // Сохраняем файл после успешной загрузки
                statusDiv.text(response.message).css('color', 'green');
                processButton.show(); // Показываем кнопку обработки

                // Отображаем изображение
                const imagePath = response.image_path;
                console.log("imagePath:", imagePath);
                imageContainer.html(`<img src="/${imagePath}" alt="Processed Image" />`);
                checkboxContainer.show(); // Показываем флажки
            },
            error: function(xhr, status, error) {
                const response = JSON.parse(xhr.responseText);
                statusDiv.text(response.detail).css('color', 'red');
            }
        });
    }

    // Обработчик клика по кнопке для обработки изображения
    $(document).on('click', '#process-image', function() {
        if (!uploadedFile) return;

        // Получаем выбранные флажки
        const selectedOptions = [];
        $('#checkbox-container input').each(function() {
            // Если флажок выбран, добавляем его значение, иначе добавляем "off"
            if ($(this).prop('checked')) {
                selectedOptions.push($(this).val()); // Получаем значение выбранного флажка
            } else {
                selectedOptions.push('off'); // Проставляем 'off' для невыбранных флажков
            }
        });

        // Проверка, что данные не пусты
        if (selectedOptions.length === 0) {
            alert("Выберите хотя бы один флажок!");
            return;
        }

        // Сериализация в JSON и отправка
        const optionsJson = JSON.stringify(selectedOptions);

        // Отправляем запрос на сервер для обработки изображения с выбранными опциями
        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("options", optionsJson); // Отправляем выбранные флажки как строку JSON

        $.ajax({
            url: '/process/', // Путь для обработки изображения
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                statusDiv.text(response.message).css('color', 'green');
                // Дополнительные действия после успешной обработки
<!--                imageContainer.html(`<img src="${response.image_path}" alt="Processed Image" />`);-->
<!--                console.log("Обработанные флажки:", response.results); // Логируем результаты обработки-->
                // Редирект на другую страницу после успешной обработки
                window.location.href = "/processed_image"; // Переход на страницу с результатами обработки
            },
            error: function(xhr, status, error) {
                const response = JSON.parse(xhr.responseText);
                statusDiv.text(response.detail).css('color', 'red');
            }
        });
    });
});