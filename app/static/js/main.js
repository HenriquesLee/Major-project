document.addEventListener('DOMContentLoaded', function() {
    // Set up mode change handlers
    ['image', 'video', 'audio'].forEach(domain => {
        const modeSelect = document.getElementById(`${domain}-mode`);
        const examplesDiv = document.getElementById(`${domain}-examples`);
        
        modeSelect.addEventListener('change', function() {
            if (this.value === 'zero-shot') {
                examplesDiv.classList.add('d-none');
            } else {
                examplesDiv.classList.remove('d-none');
            }
        });
    });
});

async function processFile(domain) {
    const mode = document.getElementById(`${domain}-mode`).value;
    const file = document.getElementById(`${domain}-input`).files[0];
    const examplesInput = document.getElementById(`${domain}-examples-input`);
    
    if (!file) {
        alert('Please select a file to analyze');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    if (mode === 'one-shot' && examplesInput.files.length !== 1) {
        alert('Please select exactly one example file for one-shot learning');
        return;
    }

    if (mode === 'few-shot' && examplesInput.files.length < 2) {
        alert('Please select multiple example files for few-shot learning');
        return;
    }

    if (mode === 'one-shot') {
        formData.append('example', examplesInput.files[0]);
    } else if (mode === 'few-shot') {
        Array.from(examplesInput.files).forEach(file => {
            formData.append('examples', file);
        });
    }

    try {
        const response = await fetch(`/api/predict/${domain}/${mode}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'An error occurred');
        }
    } catch (error) {
        displayResults({ error: error.message });
    }
}

function displayResults(results) {
    const resultsElement = document.getElementById('results');
    resultsElement.textContent = JSON.stringify(results, null, 2);
}

function processImage() {
    processFile('image');
}

function processVideo() {
    processFile('video');
}

function processAudio() {
    processFile('audio');
} 