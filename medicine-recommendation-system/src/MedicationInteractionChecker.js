import React, { useState } from "react";
import "./MedicationInteractionChecker.css";

const medicationList = {
  aspirin: ["ibuprofen", "warfarin", "naproxen", "methotrexate"],
  acetaminophen: ["warfarin", "alcohol", "isoniazid"],
  ibuprofen: ["aspirin", "naproxen", "lisinopril", "methotrexate"],
  warfarin: ["aspirin", "acetaminophen", "ibuprofen", "antibiotics", "NSAIDs"],
  naproxen: ["aspirin", "ibuprofen", "warfarin", "lisinopril"],
  amoxicillin: ["probenecid", "methotrexate"],
  metformin: ["alcohol", "corticosteroids", "dextrose"],
  lisinopril: ["ibuprofen", "naproxen", "potassium-sparing diuretics"],
  statins: ["grapefruit juice", "niacin", "fibrates"],
  clopidogrel: ["omeprazole", "esomeprazole"],
  levothyroxine: ["calcium", "iron supplements", "soy"],
  prednisone: ["NSAIDs", "vaccines", "diuretics"],
  diazepam: ["alcohol", "opioids", "cimetidine"],
  metoprolol: ["verapamil", "digoxin", "antiarrhythmics"],
  fluoxetine: ["MAO inhibitors", "triptans", "warfarin"],
  sertraline: ["NSAIDs", "alcohol", "antiplatelet agents"],
  carbamazepine: ["warfarin", "oral contraceptives", "CYP3A4 inducers"],
  azithromycin: ["warfarin", "digoxin"],
  lisinopril: ["diuretics", "potassium supplements"],
  fentanyl: ["other opioids", "CYP3A4 inhibitors", "muscle relaxants"],
  tramadol: ["SSRIs", "MAO inhibitors", "other CNS depressants"],
  codeine: ["alcohol", "benzodiazepines", "antidepressants"],
  alprazolam: ["opioids", "CYP3A4 inhibitors", "alcohol"],
  duloxetine: ["NSAIDs", "anticoagulants", "triptans"],
  methylphenidate: ["MAO inhibitors", "antidepressants"],
  bupropion: ["other antidepressants", "seizure medications", "MAO inhibitors"],
  atorvastatin: ["grapefruit juice", "niacin"],
  sotalol: ["digoxin", "antiarrhythmics"],
  potassiumchloride: ["ACE inhibitors", "potassium-sparing diuretics"],
  nitroglycerin: ["sildenafil", "other nitrates"],
  trazodone: ["MAO inhibitors", "CNS depressants"],
  hydroxyzine: ["CNS depressants", "opioids"],
  oxycodone: ["alcohol", "benzodiazepines", "CYP3A4 inhibitors"],
  clonazepam: ["opioids", "alcohol", "CYP3A4 inhibitors"],
  valproate: ["aspirin", "lamotrigine"],
  quetiapine: ["CNS depressants", "antihypertensives"],
  risperidone: ["CNS depressants", "antihypertensives"],
  lisdexamfetamine: ["MAO inhibitors", "tricyclic antidepressants"],
  methotrexate: ["NSAIDs", "sulfonamides", "penicillins"],
  furosemide: ["NSAIDs", "aminoglycosides"],
  prednisone: ["vaccines", "antibiotics"],
  gentamicin: ["furosemide", "vancomycin"],
};

const MedicationInteractionChecker = () => {
  const [medication1, setMedication1] = useState("");
  const [medication2, setMedication2] = useState("");
  const [interaction, setInteraction] = useState("");

  const checkInteraction = () => {
    if (
      medicationList[medication1]?.includes(medication2) ||
      medicationList[medication2]?.includes(medication1)
    ) {
      setInteraction(
        `Warning: ${medication1} may interact with ${medication2}`
      );
    } else {
      setInteraction(
        `No known interaction between ${medication1} and ${medication2}`
      );
    }
  };

  return (
    <div className="interaction-checker">
      <h3>Check for Medication Interactions</h3>
      <input
        type="text"
        placeholder="Enter first medication"
        className="input"
        value={medication1}
        onChange={(e) => setMedication1(e.target.value.toLowerCase())}
      />
      <input
        type="text"
        placeholder="Enter second medication"
        className="input"
        value={medication2}
        onChange={(e) => setMedication2(e.target.value.toLowerCase())}
      />
      <button className="button" onClick={checkInteraction}>
        Check Interaction
      </button>
      {interaction && <p className="interaction-message">{interaction}</p>}
    </div>
  );
};

export default MedicationInteractionChecker;
