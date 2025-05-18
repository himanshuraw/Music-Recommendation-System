const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const morgan = require('morgan')

const userRoute = require('./routes/user.route')
const mlRoute = require('./routes/ml.route')
const likeRoute = require('./routes/like.route')

require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());
app.use(morgan('dev'));

app.use('/user', userRoute);
app.use('/ml', mlRoute);
app.use('/like', likeRoute);

if (!process.env.MONGODB_URI) {
  console.error('❌  Missing MONGODB_URI environment variable');
  process.exit(1);
}
mongoose.connect(process.env.MONGODB_URI);
const db = mongoose.connection;

db.on('error', (error) => console.error(error));
db.once('open', () => console.log('Connected to Database'));


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
})
