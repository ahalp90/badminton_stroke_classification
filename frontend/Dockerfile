# Production-only Dockerfile for the frontend.
# NOT used for local dev. In dev, docker-compose runs the Vite dev server
# directly from the node:20-alpine image with hot reload.
#
# This does a two-stage build:
#   Stage 1 (builder): installs Node deps and runs `npm run build`
#   Stage 2: copies the built files into an nginx container to serve them
#
# The nginx routing config (which proxies /api/* to the backend) is mounted
# in at runtime via docker-compose.prod.yml. See nginx/nginx.conf.

FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
