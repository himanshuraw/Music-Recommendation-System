const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const userRoute = require('./routes/user.route')

require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());

app.use('/user', userRoute)

mongoose.connect(process.env.MONGODB_URI);
const db = mongoose.connection;

db.on('error', (error) => console.error(error));
db.once('open', () => console.log('Connected to Database'));


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
})
