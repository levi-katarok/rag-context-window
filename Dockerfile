FROM node:20-slim
WORKDIR /app
COPY package.json .
RUN npm install
COPY rag-context.ts .
CMD ["npx", "tsx", "rag-context.ts"]
