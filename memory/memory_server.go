// [Memory]: This module manages decentralized user memory for AGI.
package main

import (
	"context"
	"log"
	"net"
	"sync"

	pb "github.com/MyndScript/AGI/memory/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/reflection"
)

type UserMemory struct {
	mu      sync.RWMutex
	storage map[string]map[string]string // userID -> key -> value
}

// NewUserMemory creates a new UserMemory instance.
func NewUserMemory() *UserMemory {
	return &UserMemory{
		storage: make(map[string]map[string]string),
	}
}

// server implements the UserMemoryServiceServer gRPC interface.
type server struct {
	pb.UnimplementedUserMemoryServiceServer
	mem *UserMemory
}

// Set stores a key-value pair for a user.
func (s *server) Set(_ context.Context, req *pb.SetRequest) (*pb.SetResponse, error) {
	s.mem.mu.Lock()
	defer s.mem.mu.Unlock()
	if _, exists := s.mem.storage[req.UserId]; !exists {
		s.mem.storage[req.UserId] = make(map[string]string)
	}
	s.mem.storage[req.UserId][req.Key] = req.Value
	return &pb.SetResponse{Success: true}, nil
}

// Get retrieves a value for a user's key.
func (s *server) Get(_ context.Context, req *pb.GetRequest) (*pb.GetResponse, error) {
	s.mem.mu.RLock()
	defer s.mem.mu.RUnlock()
	if userData, exists := s.mem.storage[req.UserId]; exists {
		val, ok := userData[req.Key]
		return &pb.GetResponse{Value: val, Found: ok}, nil
	}
	return &pb.GetResponse{Value: "", Found: false}, nil
}

// GetUserContext returns all key-value pairs for a user as a string.
func (s *server) GetUserContext(_ context.Context, req *pb.UserContextRequest) (*pb.UserContextResponse, error) {
	s.mem.mu.RLock()
	defer s.mem.mu.RUnlock()
	context := ""
	if userData, exists := s.mem.storage[req.UserId]; exists {
		for k, v := range userData {
			context += k + ": " + v + "\n"
		}
	}
	return &pb.UserContextResponse{Context: context}, nil
}

func main() {
	lis, err := net.Listen("tcp", "127.0.0.1:50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	creds, err := credentials.NewServerTLSFromFile("cert.pem", "key.pem")
	if err != nil {
		log.Fatalf("failed to load TLS credentials: %v", err)
	}
	grpcServer := grpc.NewServer(grpc.Creds(creds))
	mem := NewUserMemory()
	pb.RegisterUserMemoryServiceServer(grpcServer, &server{mem: mem})
	reflection.Register(grpcServer)
	log.Println("Memory gRPC server running securely on 127.0.0.1:50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
