# ───────────────────────────────
# 1. Build stage
# ───────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json vite.config.js ./
COPY .env .env
RUN npm ci

COPY src/ src/
COPY public/ public/
COPY index.html .

RUN npm run build


# ───────────────────────────────
# 2. Serve stage
# ───────────────────────────────
FROM nginx:stable-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy your custom nginx config
COPY nginx.conf /etc/nginx/conf.d/

# Copy built frontend from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose the port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
