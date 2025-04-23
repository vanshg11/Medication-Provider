import React, { useState, useEffect } from "react";
import "./App.css";
import medicineData from "./medicineData.json";
import Calendar from "react-calendar";
import "react-calendar/dist/Calendar.css";
import MedicationInteractionChecker from "./MedicationInteractionChecker";
import ChatConsultation from "./ChatConsultation";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [prescriptions, setPrescriptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [reminderTime, setReminderTime] = useState("");
  const [reminders, setReminders] = useState([]);
  const [history, setHistory] = useState([]);
  const [date, setDate] = useState(new Date());
  const [weight, setWeight] = useState("");
  const [height, setHeight] = useState("");
  const [age, setAge] = useState("");
  const [dosage, setDosage] = useState("");
  const [profile, setProfile] = useState({ name: "", email: "" });
  const [sideEffects, setSideEffects] = useState("");
  const [personalRecommendations, setPersonalRecommendations] = useState([]);
  const [progress, setProgress] = useState(0); // Initialize progress state
  const [progressLog, setProgressLog] = useState([]);
  const [symptoms, setSymptoms] = useState("");
  const [suggestedConditions, setSuggestedConditions] = useState([]);
  const [mood, setMood] = useState("");
  const [wellnessLog, setWellnessLog] = useState([]);
  const [savedSideEffects, setSavedSideEffects] = useState("");
  const handleSaveSideEffects = () => {
    // Save the side effects
    setSavedSideEffects(sideEffects);
    alert("Side effects saved successfully!");
  };
  const symptomConditionsMap = {
    headache: ["Migraine", "Tension Headache", "Sinusitis"],
    dizziness: ["Vertigo", "Anemia", "Low Blood Pressure"],
    fever: ["Flu", "COVID-19", "Infection"],
    fatigue: [
      "Chronic Fatigue Syndrome",
      "Anemia",
      "Sleep Deprivation",
      "Depression",
    ],
    cough: ["Common Cold", "COVID-19", "Bronchitis", "Asthma"],
    soreThroat: ["Strep Throat", "Viral Infection", "Tonsillitis"],
    runnyNose: ["Common Cold", "Sinus Infection", "Allergy", "Flu"],
    musclePain: ["Fibromyalgia", "Muscle Strain", "Arthritis"],
    chills: ["Malaria", "Flu", "Cold Exposure"],
    chestPain: [
      "Heart Attack",
      "Gastroesophageal Reflux Disease (GERD)",
      "Panic Attack",
    ],
    shortnessOfBreath: ["Asthma", "Pneumonia", "Heart Disease", "COVID-19"],
    abdominalPain: [
      "Indigestion",
      "Appendicitis",
      "Irritable Bowel Syndrome (IBS)",
      "Gastritis",
    ],
    nausea: ["Morning Sickness", "Food Poisoning", "Migraine", "Stomach Virus"],
    vomiting: ["Food Poisoning", "Gastric Flu", "Migraine", "Pregnancy"],
    insomnia: ["Anxiety", "Depression", "Sleep Apnea", "Chronic Pain"],
    rash: ["Allergic Reaction", "Eczema", "Psoriasis", "Shingles"],
    swollenLymphNodes: ["Infection", "Lymphoma", "Mononucleosis"],
    sweating: ["Hyperhidrosis", "Anxiety", "Menopause", "Thyroid Issues"],
    backPain: ["Sciatica", "Muscle Strain", "Herniated Disc", "Osteoarthritis"],
    frequentUrination: [
      "Urinary Tract Infection (UTI)",
      "Diabetes",
      "Prostate Issues",
    ],
    dizzinessOnStanding: [
      "Orthostatic Hypotension",
      "Dehydration",
      "Blood Pressure Issues",
    ],
    blurryVision: [
      "Diabetic Retinopathy",
      "Glaucoma",
      "Macular Degeneration",
      "Migraines",
    ],
    earPain: ["Ear Infection", "TMJ Disorder", "Sinus Infection"],
    hairLoss: ["Alopecia", "Thyroid Disorder", "Stress", "Vitamin Deficiency"],
    hotFlashes: ["Menopause", "Thyroid Imbalance", "Anxiety"],
    moodSwings: [
      "Premenstrual Syndrome (PMS)",
      "Bipolar Disorder",
      "Thyroid Imbalance",
      "Depression",
    ],
    memoryLoss: [
      "Dementia",
      "Alzheimer's Disease",
      "Vitamin B12 Deficiency",
      "Stress",
    ],
    dryMouth: ["Dehydration", "Medication Side Effects", "Diabetes"],
    difficultySwallowing: [
      "Dysphagia",
      "Stroke",
      "Esophageal Cancer",
      "Throat Infection",
    ],
    legCramps: [
      "Dehydration",
      "Low Potassium",
      "Poor Circulation",
      "Pregnancy",
    ],
    coldHandsFeet: ["Raynaud's Disease", "Anemia", "Hypothyroidism"],
    increasedThirst: ["Diabetes", "Dehydration", "Kidney Disease"],
    increasedHunger: ["Diabetes", "Hyperthyroidism", "Pregnancy"],
    jointPain: ["Osteoarthritis", "Rheumatoid Arthritis", "Gout", "Injury"],
    hiccups: [
      "Acid Reflux",
      "Stomach Irritation",
      "Nervousness",
      "Brain Injury",
    ],
    tinglingInLimbs: [
      "Neuropathy",
      "Carpal Tunnel Syndrome",
      "Vitamin B12 Deficiency",
    ],
    chillsWithoutFever: ["Hypoglycemia", "Low Blood Pressure", "Sepsis"],
    skinSensitivity: ["Fibromyalgia", "Diabetes", "Multiple Sclerosis"],
    excessiveTiredness: ["Chronic Fatigue Syndrome", "Anemia", "Depression"],
    itchySkin: ["Eczema", "Allergic Reaction", "Psoriasis", "Dry Skin"],
    decreasedAppetite: [
      "Depression",
      "Infection",
      "Cancer",
      "Thyroid Disorders",
    ],
    heartPalpitations: [
      "Anxiety",
      "Arrhythmia",
      "Hyperthyroidism",
      "Caffeine Overload",
    ],
    excessiveSweating: [
      "Hyperhidrosis",
      "Menopause",
      "Thyroid Disorders",
      "Anxiety",
    ],
    lossOfTasteSmell: ["COVID-19", "Sinus Infection", "Flu", "Allergies"],
    sensitivityToLight: ["Migraine", "Conjunctivitis", "Vertigo"],
    swollenAnkles: ["Edema", "Heart Failure", "Kidney Disease", "Pregnancy"],
    frequentHeadaches: [
      "Migraine",
      "Tension Headache",
      "Dehydration",
      "Sinus Infection",
    ],
    coldSweats: ["Panic Attack", "Hypoglycemia", "Heart Attack", "Stress"],
    difficultyBreathing: ["Asthma", "COPD", "COVID-19", "Anxiety"],
    nailChanges: ["Fungal Infection", "Thyroid Problems", "Iron Deficiency"],
    hotHandsFeet: ["Hyperthyroidism", "Fever", "Raynaud's Disease"],
    crackedLips: ["Dehydration", "Vitamin Deficiency", "Cold Weather Exposure"],
    dizzinessAfterEating: [
      "Postprandial Hypotension",
      "Food Intolerances",
      "Hypoglycemia",
    ],
  };

  const handleSearch = () => {
    setLoading(true);
    setResults([]);

    setTimeout(() => {
      const foundMedicines = medicineData[query.toLowerCase()] || [];
      setResults(foundMedicines);
      setLoading(false);

      setHistory([
        ...history,
        { query: query, timestamp: new Date().toLocaleString() },
      ]);

      // Update recommendations based on the query
      updateRecommendations(query);
    }, 1000);
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const newPrescription = {
        name: file.name,
        fileUrl: URL.createObjectURL(file),
      };
      setPrescriptions([...prescriptions, newPrescription]);
    }
  };

  const handleAddReminder = () => {
    if (reminderTime) {
      const newReminder = {
        time: reminderTime,
        id: Date.now(),
      };
      setReminders([...reminders, newReminder]);
      setReminderTime("");
    }
  };

  useEffect(() => {
    const now = new Date();
    reminders.forEach((reminder) => {
      const reminderDate = new Date(reminder.time);
      if (reminderDate > now) {
        const timeout = reminderDate - now;
        setTimeout(() => {
          alert("Reminder: It's time to take your medication!");
        }, timeout);
      }
    });
  }, [reminders]);

  const handleDosageCalculation = () => {
    if (weight && height && age) {
      const bsa = Math.sqrt((height * weight) / 3600);
      const mr = 0.1 * age + 0.5;
      const calculatedDosage = (
        (weight * bsa * mr + 10 * age) /
        (age * 1.2)
      ).toFixed(2);
      setDosage(calculatedDosage);
    }
  };

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfile({ ...profile, [name]: value });
  };

  const updateRecommendations = (query) => {
    // Sample recommendation logic
    const recommendations = medicineData[query.toLowerCase()] || [];
    setPersonalRecommendations(recommendations.map((m) => m.name));
  };
  const handleProgressChange = (e) => {
    let value = parseInt(e.target.value);
    if (value >= 0 && value <= 100) {
      setProgress(value);
      const newLogEntry = {
        percentage: value,
        timestamp: new Date().toLocaleString(),
      };
      setProgressLog([...progressLog, newLogEntry]);
      if (value === 100) {
        alert("Congratulations! You've completed your medication course.");
      }
    }
  };

  // New feature: Export search history as CSV
  const exportHistoryToCSV = () => {
    const csvContent = `data:text/csv;charset=utf-8,Query,Date\n${history
      .map((entry) => `${entry.query},${entry.timestamp}`)
      .join("\n")}`;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "search_history.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const handleSymptomCheck = () => {
    if (symptoms.trim() === "") {
      alert("Please enter at least one symptom to check conditions.");
      return;
    }

    const inputSymptoms = symptoms.toLowerCase().split(", ");
    const matchedConditions = new Set();

    inputSymptoms.forEach((symptom) => {
      if (symptomConditionsMap[symptom]) {
        symptomConditionsMap[symptom].forEach((condition) =>
          matchedConditions.add(condition)
        );
      }
    });

    if (matchedConditions.size === 0) {
      alert("No conditions found for the entered symptoms.");
    } else {
      setSuggestedConditions([...matchedConditions]);
    }
  };
  const handleClearConditions = () => {
    setSuggestedConditions([]);
    setSymptoms("");
  };
  const handleMoodLog = () => {
    if (mood) {
      const newLogEntry = {
        mood: mood,
        timestamp: new Date().toLocaleString(),
      };
      setWellnessLog([...wellnessLog, newLogEntry]);
      setMood("");
    }
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Medicine Recommendation System</h1>
      </header>

      <div className="search-container">
        <input
          type="text"
          placeholder="Enter illness or weakness"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="search-input"
        />
        <button onClick={handleSearch} className="search-button">
          Search
        </button>
      </div>

      {loading && <p className="loading">Searching...</p>}

      {!loading && results.length > 0 ? (
        <ul className="medicine-list">
          {results.map((medicine, index) => (
            <li key={index} className="medicine-item">
              <strong>{medicine.name}</strong> - {medicine.dose}
            </li>
          ))}
        </ul>
      ) : (
        !loading && <p className="no-results">No suggestions found</p>
      )}

      <div className="upload-section">
        <h2>Upload Prescription</h2>
        <input
          type="file"
          onChange={handleFileUpload}
          className="upload-input"
        />
      </div>

      {prescriptions.length > 0 && (
        <div className="prescription-list">
          <h3>Uploaded Prescriptions</h3>
          <ul>
            {prescriptions.map((prescription, index) => (
              <li key={index}>
                <a
                  href={prescription.fileUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {prescription.name}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="reminder-section">
        <h2>Medication Reminder</h2>
        <input
          type="datetime-local"
          value={reminderTime}
          onChange={(e) => setReminderTime(e.target.value)}
          className="reminder-input"
        />
        <button onClick={handleAddReminder} className="reminder-button">
          Set Reminder
        </button>
        {reminders.length > 0 && (
          <div className="reminder-list">
            <h3>Upcoming Reminders</h3>
            <ul>
              {reminders.map((reminder) => (
                <li key={reminder.id}>
                  {new Date(reminder.time).toLocaleString()}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="calendar-section">
        <h2>Medication Tracking Calendar</h2>
        <Calendar onChange={setDate} value={date} />
      </div>

      <div className="dosage-calculator">
        <h2>Medication Dosage Calculator</h2>
        <input
          type="number"
          placeholder="Enter weight (kg)"
          value={weight}
          onChange={(e) => setWeight(e.target.value)}
          className="dosage-input"
        />
        <input
          type="number"
          placeholder="Enter height (cm)"
          value={height}
          onChange={(e) => setHeight(e.target.value)}
          className="dosage-input"
        />
        <input
          type="number"
          placeholder="Enter age (years)"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          className="dosage-input"
        />
        <button onClick={handleDosageCalculation} className="dosage-button">
          Calculate Dosage
        </button>
        {dosage && (
          <p className="dosage-result">Recommended Dosage: {dosage} mg</p>
        )}
      </div>

      <div className="history-section">
        <h2>Search History</h2>
        {history.length > 0 ? (
          <ul>
            {history.map((entry, index) => (
              <li key={index}>
                <strong>Query:</strong> {entry.query} - <strong>Date:</strong>{" "}
                {entry.timestamp}
              </li>
            ))}
          </ul>
        ) : (
          <p>No search history available</p>
        )}
      </div>

      <div className="side-effects-section">
        <h2>Track Side Effects</h2>
        <textarea
          placeholder="Enter any side effects experienced..."
          value={sideEffects}
          onChange={(e) => setSideEffects(e.target.value)}
          className="side-effects-input"
        />
        <button
          onClick={handleSaveSideEffects}
          className="save-side-effects-button"
        >
          Save Side Effects
        </button>

        {savedSideEffects && (
          <div className="saved-side-effects">
            <h3>Saved Side Effects:</h3>
            <p>{savedSideEffects}</p>
          </div>
        )}
      </div>

      <div className="recommendations-section">
        <h2>Personalized Medication Recommendations</h2>
        {personalRecommendations.length > 0 ? (
          <ul>
            {personalRecommendations.map((med, index) => (
              <li key={index}>{med}</li>
            ))}
          </ul>
        ) : (
          <p>No recommendations available</p>
        )}
      </div>

      <div className="interaction-checker">
        <h2>Medication Interaction Checker</h2>
        <MedicationInteractionChecker />
      </div>

      <div className="chat-consultation">
        <h2>Notes for Self</h2>
        <ChatConsultation />
      </div>
      <div className="progress-tracker">
        <h2>Medication Progress Tracker</h2>
        <input
          type="number"
          min="0"
          max="100"
          placeholder="Enter progress %"
          value={progress}
          onChange={handleProgressChange}
          className="progress-input"
        />
        <p className="progress-display">Current Progress: {progress}%</p>
      </div>

      {/* Export Search History as CSV */}
      <div className="export-csv">
        <h2>Export Search History</h2>
        <button onClick={exportHistoryToCSV} className="export-button">
          Download CSV
        </button>
      </div>
      <div className="symptom-checker">
        <h2>Symptom Checker</h2>
        <input
          type="text"
          placeholder="Enter symptoms (e.g., headache, dizziness)"
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          className="symptom-input"
        />
        <button onClick={handleSymptomCheck} className="check-symptoms-button">
          Check Conditions
        </button>
        <button
          onClick={handleClearConditions}
          className="clear-conditions-button"
        >
          Clear Conditions
        </button>

        {suggestedConditions.length > 0 && (
          <div className="conditions-list">
            <h3>Suggested Conditions:</h3>
            <ul>
              {suggestedConditions.map((condition, index) => (
                <li key={index}>{condition}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="wellness-tracker">
        <h2>Mood and Wellness Tracker</h2>
        <input
          type="text"
          placeholder="Enter your mood (e.g., Happy, Stressed)"
          value={mood}
          onChange={(e) => setMood(e.target.value)}
          className="mood-input"
        />
        <button onClick={handleMoodLog} className="mood-log-button">
          Log Mood
        </button>
        {wellnessLog.length > 0 && (
          <div className="mood-log">
            <h3>Mood History:</h3>
            <ul>
              {wellnessLog.map((log, index) => (
                <li
                  key={index}
                  className={`mood-item mood-${log.mood.toLowerCase()}`}
                >
                  <strong>Mood:</strong> {log.mood} - <strong>Date:</strong>{" "}
                  {log.timestamp}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="disease-prediction-container">
        <p className="disease-prediction-text">
          Our Personalized AI Health Advisor <br />
          <button
            onClick={() => window.open("http://localhost:8502/", "_blank")}
            className="disease-prediction-button"
          >
            Click here to get started
          </button>
        </p>
      </div>
    </div>
  );
}

export default App;
