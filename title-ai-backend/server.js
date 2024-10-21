const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');
const authRoutes = require('./routes/Auth');
const axios = require('axios');
require('dotenv').config();

const app = express();

// Middleware
app.use(express.json());
app.use(cors({
  origin: '*', // Allow all origins (use for development only)
  methods: 'GET,POST',
  allowedHeaders: 'Content-Type,Authorization',
  credentials: true
}));


// Connect to MongoDB
connectDB();

// Authentication routes
app.use('/api', authRoutes);

// Flask API URL for title predictions
const ML_API_URL = 'http://localhost:5001/api/predict';

// Route for checking title
app.post('/api/check-title', async (req, res) => {
  const { title } = req.body;

  if (!title) {
    return res.status(400).json({ message: 'Title is required' });
  }

  try {
    const flaskResponse = await axios.post(ML_API_URL, { title });
    return res.json(flaskResponse.data);
  } catch (error) {
    console.error('Error connecting to Flask API:', error);
    return res.status(500).json({ message: 'Error processing request' });
  }
});

// Start server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
