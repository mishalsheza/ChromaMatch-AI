const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

/**
 * Checks the connection to the backend health endpoint.
 * @returns {Promise<object>} Status check result.
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`API health check failed with status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Health check error:", error);
    throw error;
  }
}

/**
 * Uploads a selfie image to be analyzed by the ShadeSense pipeline.
 * @param {File} imageFile - The file to upload.
 * @returns {Promise<object>} The detailed analysis response from the backend.
 */
export async function analyzeImage(imageFile, options = {}) {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    
    if (options.manualSampling && options.samples) {
      formData.append("manual_sampling", "true");
      formData.append("sampling_points", JSON.stringify(options.samples.map(s => ({
        x: s.originalCoords.x,
        y: s.originalCoords.y,
        label: s.label
      }))));
    }

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Analysis failed with status: ${response.status}`);
    }

    return data;
  } catch (error) {
    console.error("Analysis API error:", error);
    throw error;
  }
}
