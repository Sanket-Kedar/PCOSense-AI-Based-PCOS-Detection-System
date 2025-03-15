document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent form submission
    const formData = new FormData();
    const fileInput = document.getElementById("imageUpload");
    formData.append("image", fileInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        const resultDiv = document.getElementById("result");
        if (response.ok) {
            resultDiv.innerHTML = `<p>Prediction: <strong>${result.prediction}</strong></p>`;
        } else {
            resultDiv.innerHTML = `<p style="color:red;">Error: ${result.error}</p>`;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `<p style="color:red;">An error occurred while processing your request.</p>`;
    }
});
