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

type Moment struct {
	ID        string
	UserID    string
	Summary   string
	Emotion   string
	Glyph     string
	Tags      []string
	Timestamp int64
	Embedding string
}

type UserMemory struct {
	mu      sync.RWMutex
	storage map[string]map[string]string // userID -> key -> value
	moments map[string][]*Moment         // userID -> moments
}

// NewUserMemory creates a new UserMemory instance.
func NewUserMemory() *UserMemory {
	return &UserMemory{
		storage: make(map[string]map[string]string),
		moments: make(map[string][]*Moment),
	}
}

// server implements the UserMemoryServiceServer gRPC interface.
type server struct {
	pb.UnimplementedUserMemoryServiceServer
	mem *UserMemory
}

// SetMoment stores a structured moment for a user.
func (s *server) SetMoment(_ context.Context, req *pb.SetMomentRequest) (*pb.SetResponse, error) {
	s.mem.mu.Lock()
	defer s.mem.mu.Unlock()
	m := req.GetMoment()
	moment := &Moment{
		ID:        m.GetId(),
		UserID:    m.GetUserId(),
		Summary:   m.GetSummary(),
		Emotion:   m.GetEmotion(),
		Glyph:     m.GetGlyph(),
		Tags:      m.GetTags(),
		Timestamp: m.GetTimestamp(),
		Embedding: m.GetEmbedding(),
	}
	s.mem.moments[moment.UserID] = append(s.mem.moments[moment.UserID], moment)
	return &pb.SetResponse{Success: true}, nil
}

// GetMoments retrieves moments for a user, filtered by tags and time.
func (s *server) GetMoments(_ context.Context, req *pb.GetMomentsRequest) (*pb.GetMomentsResponse, error) {
	s.mem.mu.RLock()
	defer s.mem.mu.RUnlock()
	userID := req.GetUserId()
	tags := req.GetTags()
	since := req.GetSince()
	until := req.GetUntil()
	var results []*pb.Moment
	for _, m := range s.mem.moments[userID] {
		if (since == 0 || m.Timestamp >= since) && (until == 0 || m.Timestamp <= until) {
			if len(tags) == 0 || hasTags(m.Tags, tags) {
				results = append(results, &pb.Moment{
					Id:        m.ID,
					UserId:    m.UserID,
					Summary:   m.Summary,
					Emotion:   m.Emotion,
					Glyph:     m.Glyph,
					Tags:      m.Tags,
					Timestamp: m.Timestamp,
					Embedding: m.Embedding,
				})
			}
		}
	}
	return &pb.GetMomentsResponse{Moments: results}, nil
}

// SemanticSearch performs a simple semantic search over moments (stub: matches summary/glyph/tags).
func (s *server) SemanticSearch(_ context.Context, req *pb.SemanticSearchRequest) (*pb.SemanticSearchResponse, error) {
	s.mem.mu.RLock()
	defer s.mem.mu.RUnlock()
	userID := req.GetUserId()
	query := req.GetQuery()
	var results []*pb.Moment
	for _, m := range s.mem.moments[userID] {
		if contains(m.Summary, query) || contains(m.Glyph, query) || tagsContain(m.Tags, query) {
			results = append(results, &pb.Moment{
				Id:        m.ID,
				UserId:    m.UserID,
				Summary:   m.Summary,
				Emotion:   m.Emotion,
				Glyph:     m.Glyph,
				Tags:      m.Tags,
				Timestamp: m.Timestamp,
				Embedding: m.Embedding,
			})
		}
	}
	return &pb.SemanticSearchResponse{Results: results}, nil
}

// Helper functions
func hasTags(momentTags, filterTags []string) bool {
	tagSet := make(map[string]struct{})
	for _, t := range momentTags {
		tagSet[t] = struct{}{}
	}
	for _, ft := range filterTags {
		if _, ok := tagSet[ft]; !ok {
			return false
		}
	}
	return true
}

func contains(s, substr string) bool {
	return len(substr) == 0 || (len(s) > 0 && (s == substr || (len(s) >= len(substr) && (s == substr || (len(s) > len(substr) && (s[:len(substr)] == substr))))) || (len(s) > 0 && (len(substr) > 0 && (len(s) >= len(substr) && (s[len(s)-len(substr):] == substr)))))
}

func tagsContain(tags []string, query string) bool {
	for _, t := range tags {
		if t == query {
			return true
		}
	}
	return false
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
