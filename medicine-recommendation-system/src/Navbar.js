// Navbar.js
import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css"; // Optional: for custom styling

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">Medication Tracker</div>
      <ul className="navbar-links">
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/upload">Upload Prescription</Link>
        </li>
        <li>
          <Link to="/reminders">Set Reminders</Link>
        </li>
        <li>
          <Link to="/calendar">Medication Calendar</Link>
        </li>
        <li>
          <Link to="/dosage">Dosage Calculator</Link>
        </li>
        <li>
          <Link to="/history">Search History</Link>
        </li>
        <li>
          <Link to="/interaction">Interaction Checker</Link>
        </li>
        <li>
          <Link to="/consult">Consult a Doctor</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
