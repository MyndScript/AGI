// memory_server.go
// memory_server.go
// Go gRPC server for AGI memory system
package main

import (
	"context"
	"log"
	"net"
	"sync"

	pb "./memory"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

type UserMemory struct {
	mu      sync.RWMutex
	storage map[string]map[string]string // userID -> key -> value
}

func NewUserMemory() *UserMemory {
	return &UserMemory{
		storage: make(map[string]map[string]string),
	}
}

type server struct {
	pb.UnimplementedUserMemoryServiceServer
	mem *UserMemory
}

func (s *server) Set(ctx context.Context, req *pb.SetRequest) (*pb.SetResponse, error) {
	s.mem.mu.Lock()
	defer s.mem.mu.Unlock()
	if _, exists := s.mem.storage[req.UserId]; !exists {
		s.mem.storage[req.UserId] = make(map[string]string)
	}
	s.mem.storage[req.UserId][req.Key] = req.Value
	return &pb.SetResponse{Success: true}, nil
}

func (s *server) Get(ctx context.Context, req *pb.GetRequest) (*pb.GetResponse, error) {
	s.mem.mu.RLock()
	defer s.mem.mu.RUnlock()
	if userData, exists := s.mem.storage[req.UserId]; exists {
		val, ok := userData[req.Key]
		return &pb.GetResponse{Value: val, Found: ok}, nil
	}
	return &pb.GetResponse{Value: "", Found: false}, nil
}

func (s *server) GetUserContext(ctx context.Context, req *pb.UserContextRequest) (*pb.UserContextResponse, error) {
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
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	mem := NewUserMemory()
	pb.RegisterUserMemoryServiceServer(grpcServer, &server{mem: mem})
	reflection.Register(grpcServer)
	log.Println("Memory gRPC server running on :50051")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
