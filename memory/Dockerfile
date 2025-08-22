# AGI Go Memory Server Dockerfile
FROM golang:1.24.6-alpine
WORKDIR /app
COPY . .
RUN go build -o memory_server memory_server.go
EXPOSE 50051
CMD ["./memory_server"]
