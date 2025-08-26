
# AGI Go Memory Server Dockerfile

FROM golang:1.24.6-alpine AS builder
WORKDIR /app
COPY memory_rest_server.go go.mod go.sum ./

# Build the binary for Linux explicitly
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o memory_server memory_rest_server.go
RUN ls -l /app

# Final stage: minimal runtime image
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/memory_server .
RUN ls -l /app
RUN chmod +x /app/memory_server

# Optional: install debug utilities
RUN apk add --no-cache file libc6-compat
EXPOSE 50051

# Run the binary directly
CMD ["./memory_server"]
