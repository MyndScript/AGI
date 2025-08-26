# AGI UI Dockerfile
FROM node:20
WORKDIR /app
COPY package*.json ./
RUN npm install --legacy-peer-deps
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npx", "serve", "dist", "-l", "3000"]
