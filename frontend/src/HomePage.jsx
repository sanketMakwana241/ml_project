import React,{ useState } from "react";

export default function HomePage() {
  const [formData, setFormData] = useState({
    Battery: "",
    Weight: "",
    NoOfCameras: "",
    Is5GSupport: false,
    IsDualSimSupport: false,
    ModelYear: "",
    Company: "",
    FrameRate: "",
  });

  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState("");

  const companies = ["Realme", "Samsung", "IPhone", "Vivo", "Oppo", "OnePlus"];
  const frameRates = [30, 60, 90, 120];

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSubmit = async () => {
    setError(""); 
    setPredictedPrice(null);

    // Validation
    if (
      !formData.Battery ||
      !formData.Weight ||
      !formData.NoOfCameras ||
      !formData.ModelYear ||
      !formData.Company ||
      !formData.FrameRate
    ) {
      setError("All fields are required!");
      return;
    }

    const payload = {
      Battery: parseInt(formData.Battery),
      Weight: parseInt(formData.Weight),
      NoOfCameras: parseInt(formData.NoOfCameras),
      Is5GSupport: formData.Is5GSupport,
      IsDualSimSupport: formData.IsDualSimSupport,
      ModelYear: parseInt(formData.ModelYear),
      Company: formData.Company,
      FrameRate: parseInt(formData.FrameRate),
    };

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      setPredictedPrice(data["Predicted Price"]);
    } catch (error) {
      setError("Error fetching price. Try again later.");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-lg">
        <h1 className="text-2xl font-bold text-center text-gray-700 mb-6">Phone Price Predictor</h1>

        <div className="grid grid-cols-2 gap-4">
          {/* Numeric Inputs */}
          {["Battery", "Weight", "NoOfCameras", "ModelYear"].map((field) => (
            <div key={field} className="flex flex-col">
              <label className="text-gray-600">{field}:</label>
              <input
                type="number"
                name={field}
                value={formData[field]}
                onChange={handleChange}
                className="border p-2 rounded"
              />
            </div>
          ))}

          {/* Dropdowns */}
          <div className="flex flex-col">
            <label className="text-gray-600">Company:</label>
            <select
              name="Company"
              value={formData.Company}
              onChange={handleChange}
              className="border p-2 rounded"
            >
              <option value="">Select Company</option>
              {companies.map((company) => (
                <option key={company} value={company}>
                  {company}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col">
            <label className="text-gray-600">Frame Rate:</label>
            <select
              name="FrameRate"
              value={formData.FrameRate}
              onChange={handleChange}
              className="border p-2 rounded"
            >
              <option value="">Select Frame Rate</option>
              {frameRates.map((rate) => (
                <option key={rate} value={rate}>
                  {rate} FPS
                </option>
              ))}
            </select>
          </div>

          {/* Checkboxes */}
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="Is5GSupport"
              checked={formData.Is5GSupport}
              onChange={handleChange}
              className="w-4 h-4"
            />
            <label className="text-gray-600">5G Support</label>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              name="IsDualSimSupport"
              checked={formData.IsDualSimSupport}
              onChange={handleChange}
              className="w-4 h-4"
            />
            <label className="text-gray-600">Dual SIM Support</label>
          </div>
        </div>

        {/* Error Message */}
        {error && <p className="text-red-500 mt-4 text-center">{error}</p>}

        {/* Check Price Button */}
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white p-2 rounded mt-6"
        >
          Check Price
        </button>

        {/* Display Predicted Price */}
        {predictedPrice !== null && (
          <div className="mt-6 text-center text-lg font-semibold">
            Predicted Price: <span className="text-green-600">₹{predictedPrice}</span>
          </div>
        )}
      </div>
    </div>
  );
}