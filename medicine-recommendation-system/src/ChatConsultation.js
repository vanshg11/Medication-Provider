import React, { useState } from "react";
import "./ChatConsultation.css";

const ChatConsultation = () => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const handleSendMessage = () => {
    const newMessage = {
      text: message,
      timestamp: new Date().toLocaleString(),
    };
    setChatHistory([...chatHistory, newMessage]);
    setMessage("");
  };

  return (
    <div className="chat-container">
      <h3>Start Writing, we're here to remind you to tick them up</h3>
      <div className="chat-window">
        {chatHistory.map((chat, index) => (
          <div key={index} className="chat-message">
            <strong>{chat.timestamp}:</strong> {chat.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        placeholder="Type your message..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        className="chat-input"
      />
      <button onClick={handleSendMessage} className="chat-send-button">
        Put it Up
      </button>
    </div>
  );
};

export default ChatConsultation;
