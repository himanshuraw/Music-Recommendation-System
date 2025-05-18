# 1. Base image
FROM node:18-alpine

# 2. Create app directory
WORKDIR /usr/src/app

# 3. Install dependencies
COPY package*.json ./
RUN npm ci 

# 4. Copy source
COPY . .

# 5. Expose port and set NODE_ENV
ENV NODE_ENV=production
EXPOSE 3000

# 6. Start the app
CMD ["node", "src/app.js"]