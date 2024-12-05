$(document).ready(function() {
    const uploadArea = $('#upload-area');
    const fileInput = $('#file-input');
    const statusDiv = $('#upload-status');

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
            url: '/upload/',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                statusDiv.text(response.message).css('color', 'green');
            },
            error: function(xhr, status, error) {
                const response = JSON.parse(xhr.responseText);
                statusDiv.text(response.detail).css('color', 'red');
            }
        });
    }
});
