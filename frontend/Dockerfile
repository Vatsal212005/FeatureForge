FROM node:20-alpine

WORKDIR /workspace/frontend

COPY frontend/package.json /workspace/frontend/package.json

RUN npm install

COPY frontend /workspace/frontend

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
