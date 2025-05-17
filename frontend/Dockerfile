# ───────────────────────────────
# 1. Build stage
# ───────────────────────────────
FROM node:18-alpine AS builder

# 1.1 Set workdir to /app
WORKDIR /app

# 1.2 Copy only manifest + config files
COPY package.json package-lock.json vite.config.js ./

# 1.3 Copy your env (so VITE_… vars are in the build)
COPY .env .env

# 1.4 Install deps
RUN npm ci

# 1.5 Copy the rest of your source
COPY src/ src/
COPY public/ public/
COPY index.html .

# 1.6 Build for production
RUN npm run build



# ───────────────────────────────
# 2. Serve stage
# ───────────────────────────────
FROM nginx:stable-alpine

# 2.1 Remove default conf and use our nginx.conf
RUN rm /etc/nginx/conf.d/default.conf

# 2.2 Copy your custom nginx.conf (for SPA routing)
COPY nginx.conf /etc/nginx/conf.d/

# 2.3 Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# 2.4 Expose port 80
EXPOSE 80

# 2.5 Launch Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
