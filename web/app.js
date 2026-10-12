const form = document.querySelector("#valuation-form");
const estimateButton = document.querySelector("#estimate-button");
const buttonLabel = document.querySelector(".button-label");
const formError = document.querySelector("#form-error");
const resultKicker = document.querySelector("#result-kicker");
const resultValue = document.querySelector("#result-value");
const resultUnit = document.querySelector("#result-unit");
const resetButton = document.querySelector("#reset-button");

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function clearError() {
  formError.textContent = "";
  formError.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  if (!form.reportValidity()) return;

  const formData = new FormData(form);
  const payload = {
    "X1 transaction date": Number(formData.get("transaction_date")),
    "X2 house age": Number(formData.get("house_age")),
    "X3 distance to the nearest MRT station": Number(formData.get("mrt_distance")),
    "X4 number of convenience stores": Number(formData.get("convenience_stores")),
    "X5 latitude": Number(formData.get("latitude")),
    "X6 longitude": Number(formData.get("longitude")),
  };

  estimateButton.disabled = true;
  buttonLabel.textContent = "Working it out…";
  resultKicker.textContent = "CALCULATING ESTIMATE";
  resultValue.textContent = "…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The estimate could not be calculated.");

    resultKicker.textContent = "ESTIMATED VALUE PER UNIT AREA";
    resultValue.textContent = data.predicted_price.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    resultUnit.textContent = "Dataset price units per unit area · indicative estimate";
  } catch (error) {
    resultKicker.textContent = "ESTIMATE NOT AVAILABLE";
    resultValue.innerHTML = '<span class="result-placeholder">Let’s try that<br>one more time.</span>';
    showError(error instanceof Error ? error.message : "Something went wrong. Please try again.");
  } finally {
    estimateButton.disabled = false;
    buttonLabel.textContent = "Estimate value";
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  clearError();
  resultKicker.textContent = "READY WHEN YOU ARE";
  resultValue.innerHTML = '<span class="result-dash">—</span><span class="result-placeholder">Run an estimate<br>to see a value</span>';
  resultUnit.textContent = "Price per unit area · dataset units";
});

form.addEventListener("input", clearError);
