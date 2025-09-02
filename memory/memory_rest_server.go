// AGI Memory REST Server
// Overseer-compatible: Accepts requests proxied from overseer gateway (port 8010)
// CORS restricted to overseer gateway for security
package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"
)

// Event system
type EventType string

const (
	EventMomentStored    EventType = "moment_stored"
	EventPostIngested    EventType = "post_ingested"
	EventMemorySet       EventType = "memory_set"
	EventGlobalMemorySet EventType = "global_memory_set"
)

type Event struct {
	Type      EventType
	UserID    string
	Payload   interface{}
	Timestamp int64
}

type EventHandler func(event Event)

var eventHandlers = map[EventType][]EventHandler{}

func RegisterEventHandler(eventType EventType, handler EventHandler) {
	eventHandlers[eventType] = append(eventHandlers[eventType], handler)
}

func DispatchEvent(eventType EventType, userID string, payload interface{}) {
	for _, handler := range eventHandlers[eventType] {
		handler(Event{Type: eventType, UserID: userID, Payload: payload, Timestamp: time.Now().Unix()})
	}
}

// Data models
type Moment struct {
	ID        string   `json:"id"`
	UserID    string   `json:"user_id"`
	Summary   string   `json:"summary"`
	Emotion   string   `json:"emotion"`
	Glyph     string   `json:"glyph"`
	Tags      []string `json:"tags"`
	Timestamp int64    `json:"timestamp"`
	Embedding string   `json:"embedding"`
}

type SetRequest struct {
	UserID string `json:"user_id"`
	Key    string `json:"key"`
	Value  string `json:"value"`
}

type SetResponse struct {
	Success bool `json:"success"`
}

type GetRequest struct {
	UserID string `json:"user_id"`
	Key    string `json:"key"`
}

type GetResponse struct {
	Value string `json:"value"`
	Found bool   `json:"found"`
}

type UserContextRequest struct {
	UserID string `json:"user_id"`
}

type SetMomentRequest struct {
	Moment Moment `json:"moment"`
}

type GetMomentsRequest struct {
	UserID string   `json:"user_id"`
	Tags   []string `json:"tags"`
	Since  int64    `json:"since"`
	Until  int64    `json:"until"`
}

type SemanticSearchRequest struct {
	UserID         string `json:"user_id"`
	Query          string `json:"query"`
	QueryEmbedding string `json:"query_embedding"`
	TopK           int    `json:"top_k"`
}

type ScoredMoment struct {
	Moment     Moment  `json:"moment"`
	Similarity float64 `json:"similarity"`
}

type PersonalityTraitRequest struct {
	UserID string  `json:"user_id"`
	Trait  string  `json:"trait"`
	Score  float64 `json:"score"`
}

type PersonalityIngestRequest struct {
	UserID string `json:"user_id"`
	Posts  []struct {
		PostID    string   `json:"post_id"`
		Content   string   `json:"content"`
		Timestamp int64    `json:"timestamp"`
		Tags      []string `json:"tags"`
	} `json:"posts"`
}

type Post struct {
	PostID    string   `json:"post_id"`
	UserID    string   `json:"user_id"`
	Content   string   `json:"content"`
	Timestamp int64    `json:"timestamp"`
	Tags      []string `json:"tags"`
}

// Embedding types
type IngestRequest struct {
	UserID string        `json:"user_id"`
	Posts  []interface{} `json:"posts"`
}

type IngestResponse struct {
	Result bool `json:"result"`
}

// Embedding service types
type EmbeddingRequest struct {
	Text      string `json:"text"`
	Model     string `json:"model"`
	Normalize bool   `json:"normalize"`
}

type EmbeddingResponse struct {
	Embedding  []float32 `json:"embedding"`
	Model      string    `json:"model"`
	Dimensions int       `json:"dimensions"`
	TextLength int       `json:"text_length"`
}

func main() {
	// Setup DBs
	sqliteConn, pgConn := setupDatabases()
	// Setup Gin
	r := gin.Default()
	setupCORS(r)
	registerEndpoints(r, sqliteConn, pgConn)
	registerEmbeddingEndpoints(r, sqliteConn)
	r.Run(":8001")
}

// setupDatabases initializes SQLite and Postgres connections and creates required tables
func setupDatabases() (*sql.DB, *sql.DB) {
	sqliteConn, err := sql.Open("sqlite3", "user_memory.db")
	if err != nil {
		log.Fatalf("Failed to connect to SQLite: %v", err)
	}
	pgConnStr := "postgres://postgres:Br!an0525@localhost:5432/postgres?sslmode=disable"
	pgConn, err := sql.Open("postgres", pgConnStr)
	if err != nil {
		log.Fatalf("Failed to connect to Postgres: %v", err)
	}
	_, err = pgConn.Exec(`CREATE TABLE IF NOT EXISTS global_memory (
		 user_id TEXT,
		 key TEXT,
		 value TEXT,
		 PRIMARY KEY (user_id, key)
	 )`)
	if err != nil {
		log.Fatalf("Failed to create global_memory table: %v", err)
	}
	_, err = pgConn.Exec(`CREATE TABLE IF NOT EXISTS personality_matrix (
		user_id TEXT,
		trait TEXT,
		score FLOAT,
		last_updated BIGINT,
		PRIMARY KEY (user_id, trait)
	)`)
	if err != nil {
		log.Fatalf("Failed to create personality_matrix table: %v", err)
	}
	if err != nil {
		log.Fatalf("Failed to create global_memory table: %v", err)
	}
	_, err = pgConn.Exec(`CREATE TABLE IF NOT EXISTS global_moments (
		 id TEXT PRIMARY KEY,
		 user_id TEXT,
		 summary TEXT,
		 emotion TEXT,
		 glyph TEXT,
		 tags TEXT,
		 timestamp BIGINT,
		 embedding TEXT
	 )`)
	if err != nil {
		log.Fatalf("Failed to create global_moments table: %v", err)
	}
	_, err = sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS user_memory (
		user_id TEXT,
		key TEXT,
		value TEXT,
		PRIMARY KEY (user_id, key)
	)`)
	if err != nil {
		log.Fatalf("Failed to create user_memory table: %v", err)
	}
	_, err = sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS moments (
		id TEXT PRIMARY KEY,
		user_id TEXT,
		summary TEXT,
		emotion TEXT,
		glyph TEXT,
		tags TEXT,
		timestamp BIGINT,
		embedding TEXT
	)`)
	if err != nil {
		log.Fatalf("Failed to create moments table: %v", err)
	}

	// Create embeddings table for semantic search
	_, err = sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS embeddings (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		post_id TEXT UNIQUE,
		user_id TEXT,
		text TEXT,
		embedding_json TEXT,
		model TEXT,
		dimensions INTEGER,
		created_at BIGINT,
		FOREIGN KEY(user_id) REFERENCES user_memory(user_id)
	)`)
	if err != nil {
		log.Fatalf("Failed to create embeddings table: %v", err)
	}

	// Create index for faster semantic search queries
	_, err = sqliteConn.Exec(`CREATE INDEX IF NOT EXISTS idx_embeddings_user_model ON embeddings(user_id, model)`)
	if err != nil {
		log.Fatalf("Failed to create embeddings index: %v", err)
	}

	return sqliteConn, pgConn
}

// setupCORS configures CORS middleware for Gin
func setupCORS(r *gin.Engine) {
	r.Use(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		if origin == "http://localhost:8010" || origin == "http://localhost:3000" {
			c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
		}
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})
}

// registerEndpoints registers all REST endpoints
func registerEndpoints(r *gin.Engine, sqliteConn *sql.DB, pgConn *sql.DB) {
	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "healthy",
			"service":   "memory_server",
			"port":      8001,
			"timestamp": time.Now().Format(time.RFC3339),
		})
	})

	// List all tables in SQLite/Postgres
	r.GET("/list-tables", func(c *gin.Context) {
		db := c.Query("db")
		var tables []string
		if db == "postgres" {
			rows, err := pgConn.Query("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
			if err == nil {
				defer rows.Close()
				for rows.Next() {
					var t string
					if err := rows.Scan(&t); err == nil {
						tables = append(tables, t)
					}
				}
			}
		} else {
			rows, err := sqliteConn.Query("SELECT name FROM sqlite_master WHERE type='table'")
			if err == nil {
				defer rows.Close()
				for rows.Next() {
					var t string
					if err := rows.Scan(&t); err == nil {
						tables = append(tables, t)
					}
				}
			}
		}
		c.JSON(http.StatusOK, gin.H{"tables": tables, "count": len(tables)})
	})

	// Memory endpoints
	r.POST("/set-memory", func(c *gin.Context) {
		var req SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		if req.UserID != "" && req.Key != "" {
			_, err := sqliteConn.Exec(`INSERT INTO user_memory (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value`, req.UserID, req.Key, req.Value)
			if err != nil {
				c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
				return
			}
			DispatchEvent(EventMemorySet, req.UserID, req)
			c.JSON(http.StatusOK, SetResponse{Success: true})
			return
		}
		c.JSON(http.StatusBadRequest, SetResponse{Success: false})
	})

	r.POST("/get-memory", func(c *gin.Context) {
		var req GetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, GetResponse{Found: false})
			return
		}
		var value string
		var found bool
		row := sqliteConn.QueryRow(`SELECT value FROM user_memory WHERE user_id = ? AND key = ?`, req.UserID, req.Key)
		err := row.Scan(&value)
		if err == nil {
			found = true
		} else {
			found = false
		}
		c.JSON(http.StatusOK, GetResponse{Value: value, Found: found})
	})

	r.POST("/user_context", func(c *gin.Context) {
		var req UserContextRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"context": "", "error": "Invalid payload", "details": err.Error()})
			return
		}
		rows, err := sqliteConn.Query(`SELECT key, value FROM user_memory WHERE user_id = ?`, req.UserID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"context": "", "error": err.Error()})
			return
		}
		defer rows.Close()
		context := ""
		count := 0
		for rows.Next() {
			var k, v string
			if err := rows.Scan(&k, &v); err == nil {
				context += k + ": " + v + "\n"
				count++
			}
		}
		context = "Total memory items: " + strconv.Itoa(count) + "\n" + context
		c.JSON(http.StatusOK, gin.H{"context": context})
	})

	r.POST("/dump_memory", func(c *gin.Context) {
		var req UserContextRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "bad request"})
			return
		}
		rows, err := sqliteConn.Query(`SELECT key, value FROM user_memory WHERE user_id = ?`, req.UserID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()
		mem := map[string]string{}
		for rows.Next() {
			var k, v string
			if err := rows.Scan(&k, &v); err == nil {
				mem[k] = v
			}
		}
		c.JSON(http.StatusOK, gin.H{"memory": mem, "count": len(mem)})
	})

	// Moments endpoints
	r.POST("/set_moment", func(c *gin.Context) {
		var req SetMomentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		m := req.Moment
		tagsStr := strings.Join(m.Tags, ",")
		_, err := sqliteConn.Exec(`INSERT INTO moments (id, user_id, summary, emotion, glyph, tags, timestamp, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, emotion=excluded.emotion, glyph=excluded.glyph, tags=excluded.tags, timestamp=excluded.timestamp, embedding=excluded.embedding`, m.ID, m.UserID, m.Summary, m.Emotion, m.Glyph, tagsStr, m.Timestamp, m.Embedding)
		if err != nil {
			c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
			return
		}
		DispatchEvent(EventMomentStored, m.UserID, m)
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	r.POST("/get_moments", func(c *gin.Context) {
		var req GetMomentsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"moments": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		query := "SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM moments WHERE user_id = ?"
		params := []interface{}{req.UserID}
		if len(req.Tags) > 0 {
			tagConds := []string{}
			for range req.Tags {
				tagConds = append(tagConds, "tags LIKE ?")
			}
			query += " AND (" + strings.Join(tagConds, " OR ") + ")"
			for _, tag := range req.Tags {
				params = append(params, "%"+tag+"%")
			}
		}
		if req.Since > 0 {
			query += " AND timestamp >= ?"
			params = append(params, req.Since)
		}
		if req.Until > 0 {
			query += " AND timestamp <= ?"
			params = append(params, req.Until)
		}
		rows, err := sqliteConn.Query(query, params...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"moments": nil, "error": "DB error", "details": err.Error()})
			return
		}
		defer rows.Close()
		var moments []Moment
		for rows.Next() {
			var m Moment
			var tags string
			if err := rows.Scan(&m.ID, &m.UserID, &m.Summary, &m.Emotion, &m.Glyph, &tags, &m.Timestamp, &m.Embedding); err == nil {
				m.Tags = strings.Split(tags, ",")
				moments = append(moments, m)
			}
		}
		c.JSON(http.StatusOK, gin.H{"moments": moments, "count": len(moments)})
	})

	// Global memory endpoints
	r.POST("/set-global-memory", func(c *gin.Context) {
		var req SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		_, err := pgConn.Exec(`INSERT INTO global_memory (user_id, key, value) VALUES ($1, $2, $3) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value`, req.UserID, req.Key, req.Value)
		if err != nil {
			c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
			return
		}
		DispatchEvent(EventGlobalMemorySet, req.UserID, req)
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	r.POST("/get-global-memory", func(c *gin.Context) {
		var req GetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, GetResponse{Found: false})
			return
		}
		var value string
		var found bool
		row := pgConn.QueryRow(`SELECT value FROM global_memory WHERE user_id = $1 AND key = $2`, req.UserID, req.Key)
		err := row.Scan(&value)
		if err == nil {
			found = true
		} else {
			found = false
		}
		c.JSON(http.StatusOK, GetResponse{Value: value, Found: found})
	})

	// Global moments endpoints
	r.POST("/set-global-moment", func(c *gin.Context) {
		var req SetMomentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		m := req.Moment
		_, err := pgConn.Exec(`INSERT INTO global_moments (id, user_id, summary, emotion, glyph, tags, timestamp, embedding) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, emotion=excluded.emotion, glyph=excluded.glyph, tags=excluded.tags, timestamp=excluded.timestamp, embedding=excluded.embedding`, m.ID, m.UserID, m.Summary, m.Emotion, m.Glyph, strings.Join(m.Tags, ","), m.Timestamp, m.Embedding)
		if err != nil {
			c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
			return
		}
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	r.POST("/get-global-moments", func(c *gin.Context) {
		var req GetMomentsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"moments": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		query := `SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM global_moments WHERE user_id = $1`
		args := []interface{}{req.UserID}
		rows, err := pgConn.Query(query, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"moments": nil, "error": "DB error", "details": err.Error()})
			return
		}
		defer rows.Close()
		var moments []Moment
		for rows.Next() {
			var m Moment
			var tags string
			if err := rows.Scan(&m.ID, &m.UserID, &m.Summary, &m.Emotion, &m.Glyph, &tags, &m.Timestamp, &m.Embedding); err == nil {
				m.Tags = strings.Split(tags, ",")
				moments = append(moments, m)
			}
		}
		c.JSON(http.StatusOK, gin.H{"moments": moments, "count": len(moments)})
	})

	// Personality endpoints
	r.POST("/personality/update-trait", func(c *gin.Context) {
		var req PersonalityTraitRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.UserID == "" || req.Trait == "" {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Missing user_id or trait"})
			return
		}
		now := time.Now().Unix()
		_, err := pgConn.Exec(`INSERT INTO personality_matrix (user_id, trait, score, last_updated) VALUES ($1, $2, $3, $4) ON CONFLICT(user_id, trait) DO UPDATE SET score=excluded.score, last_updated=excluded.last_updated`, req.UserID, req.Trait, req.Score, now)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"success": true})
	})

	r.POST("/personality/get-traits", func(c *gin.Context) {
		var req struct {
			UserID string `json:"user_id"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"traits": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		rows, err := pgConn.Query(`SELECT trait, score, last_updated FROM personality_matrix WHERE user_id = $1`, req.UserID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"traits": nil, "error": err.Error()})
			return
		}
		defer rows.Close()
		traits := map[string]float64{}
		lastUpdated := map[string]int64{}
		for rows.Next() {
			var trait string
			var score float64
			var updated int64
			if err := rows.Scan(&trait, &score, &updated); err == nil {
				traits[trait] = score
				lastUpdated[trait] = updated
			}
		}
		c.JSON(http.StatusOK, gin.H{"traits": traits, "last_updated": lastUpdated, "count": len(traits)})
	})

	r.POST("/personality/ingest-posts", func(c *gin.Context) {
		var req PersonalityIngestRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.UserID == "" || len(req.Posts) == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Missing user_id or posts array"})
			return
		}
		for i, post := range req.Posts {
			if post.PostID == "" || post.Content == "" || post.Timestamp == 0 {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Post at index " + strconv.Itoa(i) + " missing required fields"})
				return
			}
		}
		_, err := sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS posts (post_id TEXT PRIMARY KEY, user_id TEXT, content TEXT, timestamp BIGINT, tags TEXT)`)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "DB error", "details": err.Error()})
			return
		}
		for _, post := range req.Posts {
			tagsStr := strings.Join(post.Tags, ",")
			_, err := sqliteConn.Exec(`INSERT INTO posts (post_id, user_id, content, timestamp, tags) VALUES (?, ?, ?, ?, ?) ON CONFLICT(post_id) DO UPDATE SET content=excluded.content, timestamp=excluded.timestamp, tags=excluded.tags`, post.PostID, req.UserID, post.Content, post.Timestamp, tagsStr)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": err.Error()})
				return
			}
			DispatchEvent(EventPostIngested, req.UserID, post)
		}
		c.JSON(http.StatusOK, gin.H{"success": true, "count": len(req.Posts)})
	})

	// Facebook posts endpoints
	r.POST("/store-post", func(c *gin.Context) {
		var req struct {
			UserID    string   `json:"user_id"`
			PostID    string   `json:"post_id"`
			Content   string   `json:"content"`
			Timestamp int64    `json:"timestamp"`
			Tags      []string `json:"tags"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.UserID == "" || req.PostID == "" || req.Content == "" || req.Timestamp == 0 {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Missing required fields"})
			return
		}
		tagsStr := strings.Join(req.Tags, ",")
		_, err := sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS posts (post_id TEXT PRIMARY KEY, user_id TEXT, content TEXT, timestamp BIGINT, tags TEXT)`)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "DB error", "details": err.Error()})
			return
		}
		_, err = sqliteConn.Exec(`INSERT INTO posts (post_id, user_id, content, timestamp, tags) VALUES (?, ?, ?, ?, ?) ON CONFLICT(post_id) DO UPDATE SET content=excluded.content, timestamp=excluded.timestamp, tags=excluded.tags`, req.PostID, req.UserID, req.Content, req.Timestamp, tagsStr)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"success": true})
	})

	r.POST("/get-posts", func(c *gin.Context) {
		var req struct {
			UserID string   `json:"user_id"`
			Tags   []string `json:"tags"`
			Since  int64    `json:"since"`
			Until  int64    `json:"until"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"posts": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		query := "SELECT post_id, user_id, content, timestamp, tags FROM posts WHERE user_id = ?"
		params := []interface{}{req.UserID}
		if len(req.Tags) > 0 {
			tagConds := []string{}
			for range req.Tags {
				tagConds = append(tagConds, "tags LIKE ?")
			}
			query += " AND (" + strings.Join(tagConds, " OR ") + ")"
			for _, tag := range req.Tags {
				params = append(params, "%"+tag+"%")
			}
		}
		if req.Since > 0 {
			query += " AND timestamp >= ?"
			params = append(params, req.Since)
		}
		if req.Until > 0 {
			query += " AND timestamp <= ?"
			params = append(params, req.Until)
		}
		rows, err := sqliteConn.Query(query, params...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"posts": nil, "error": "DB error", "details": err.Error()})
			return
		}
		defer rows.Close()
		var posts []Post
		for rows.Next() {
			var p Post
			var tags string
			if err := rows.Scan(&p.PostID, &p.UserID, &p.Content, &p.Timestamp, &tags); err == nil {
				p.Tags = strings.Split(tags, ",")
				posts = append(posts, p)
			}
		}
		c.JSON(http.StatusOK, gin.H{"posts": posts, "count": len(posts)})
	})

	// Semantic search endpoint
	r.POST("/semantic_search", func(c *gin.Context) {
		var req SemanticSearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"results": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		rows, err := sqliteConn.Query(`SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM moments WHERE user_id = ?`, req.UserID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"results": nil, "error": "DB error", "details": err.Error()})
			return
		}
		defer rows.Close()
		var results []Moment
		for rows.Next() {
			var m Moment
			var tags string
			if err := rows.Scan(&m.ID, &m.UserID, &m.Summary, &m.Emotion, &m.Glyph, &tags, &m.Timestamp, &m.Embedding); err == nil {
				m.Tags = strings.Split(tags, ",")
				if strings.Contains(strings.ToLower(m.Summary), strings.ToLower(req.Query)) ||
					strings.Contains(strings.ToLower(m.Emotion), strings.ToLower(req.Query)) {
					results = append(results, m)
					continue
				}
				for _, tag := range m.Tags {
					if strings.Contains(strings.ToLower(tag), strings.ToLower(req.Query)) {
						results = append(results, m)
						break
					}
				}
			}
		}
		c.JSON(http.StatusOK, gin.H{"results": results, "count": len(results)})
	})

	// Unified get endpoint (local/global)
	r.POST("/get-memory-unified", func(c *gin.Context) {
		var req struct {
			UserID      string `json:"user_id"`
			Key         string `json:"key"`
			GlobalScope bool   `json:"global_scope"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.GlobalScope {
			if req.Key != "" {
				var value string
				row := pgConn.QueryRow(`SELECT value FROM global_memory WHERE user_id = $1 AND key = $2`, req.UserID, req.Key)
				err := row.Scan(&value)
				if err == nil {
					c.JSON(http.StatusOK, gin.H{"value": value, "found": true})
				} else {
					c.JSON(http.StatusOK, gin.H{"value": "", "found": false})
				}
				return
			} else {
				rows, err := pgConn.Query(`SELECT key, value FROM global_memory WHERE user_id = $1`, req.UserID)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
					return
				}
				defer rows.Close()
				mem := map[string]string{}
				for rows.Next() {
					var k, v string
					if err := rows.Scan(&k, &v); err == nil {
						mem[k] = v
					}
				}
				c.JSON(http.StatusOK, gin.H{"memory": mem, "count": len(mem)})
				return
			}
		}
		if req.Key != "" {
			var value string
			row := sqliteConn.QueryRow(`SELECT value FROM user_memory WHERE user_id = ? AND key = ?`, req.UserID, req.Key)
			err := row.Scan(&value)
			if err == nil {
				c.JSON(http.StatusOK, gin.H{"value": value, "found": true})
			} else {
				c.JSON(http.StatusOK, gin.H{"value": "", "found": false})
			}
			return
		}
		rows, err := sqliteConn.Query(`SELECT key, value FROM user_memory WHERE user_id = ?`, req.UserID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer rows.Close()
		mem := map[string]string{}
		for rows.Next() {
			var k, v string
			if err := rows.Scan(&k, &v); err == nil {
				mem[k] = v
			}
		}
		c.JSON(http.StatusOK, gin.H{"memory": mem, "count": len(mem)})
	})

	// Journal endpoints
	r.POST("/store-journal-entry", func(c *gin.Context) {
		var req struct {
			UserID      string                 `json:"user_id"`
			JournalData map[string]interface{} `json:"journal_data"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.UserID == "" || req.JournalData == nil {
			c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "Missing user_id or journal_data"})
			return
		}

		// Create journal_entries table if it doesn't exist
		_, err := sqliteConn.Exec(`CREATE TABLE IF NOT EXISTS journal_entries (
			id TEXT PRIMARY KEY,
			user_id TEXT,
			content TEXT,
			timestamp TEXT,
			date TEXT,
			time TEXT,
			type TEXT,
			created_at INTEGER DEFAULT (strftime('%s', 'now'))
		)`)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": "DB error", "details": err.Error()})
			return
		}

		// Insert journal entry
		_, err = sqliteConn.Exec(`INSERT INTO journal_entries (id, user_id, content, timestamp, date, time, type) 
			VALUES (?, ?, ?, ?, ?, ?, ?) 
			ON CONFLICT(id) DO UPDATE SET 
				content=excluded.content, 
				timestamp=excluded.timestamp, 
				date=excluded.date, 
				time=excluded.time, 
				type=excluded.type`,
			req.JournalData["id"], req.UserID, req.JournalData["content"],
			req.JournalData["timestamp"], req.JournalData["date"],
			req.JournalData["time"], req.JournalData["type"])

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"success": false, "error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"success": true})
	})

	r.POST("/get-journal-entries", func(c *gin.Context) {
		var req struct {
			UserID string `json:"user_id"`
			Limit  int    `json:"limit"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"entries": nil, "error": "Invalid payload", "details": err.Error()})
			return
		}
		if req.UserID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"entries": nil, "error": "Missing user_id"})
			return
		}

		limit := req.Limit
		if limit <= 0 || limit > 100 {
			limit = 10 // Default limit
		}

		rows, err := sqliteConn.Query(`SELECT id, user_id, content, timestamp, date, time, type 
			FROM journal_entries 
			WHERE user_id = ? 
			ORDER BY created_at DESC 
			LIMIT ?`, req.UserID, limit)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"entries": nil, "error": "DB error", "details": err.Error()})
			return
		}
		defer rows.Close()

		var entries []map[string]interface{}
		for rows.Next() {
			var id, userID, content, timestamp, date, time, entryType string
			if err := rows.Scan(&id, &userID, &content, &timestamp, &date, &time, &entryType); err == nil {
				entry := map[string]interface{}{
					"id":        id,
					"user_id":   userID,
					"content":   content,
					"timestamp": timestamp,
					"date":      date,
					"time":      time,
					"type":      entryType,
				}
				entries = append(entries, entry)
			}
		}

		c.JSON(http.StatusOK, gin.H{"entries": entries, "count": len(entries)})
	})
}

// Embedding functions
func generateProductionEmbedding(text string, embeddingType string) ([]float32, error) {
	// Call Python embedding service on localhost:8003
	reqBody := EmbeddingRequest{
		Text:      text,
		Model:     embeddingType,
		Normalize: true,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		log.Printf("[Embedding] Failed to marshal request: %v", err)
		return nil, err
	}

	resp, err := http.Post("http://localhost:8003/embed", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("[Embedding] Service unavailable, using fallback: %v", err)
		return generateFallbackEmbedding(text), nil
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Embedding] Failed to read response: %v", err)
		return generateFallbackEmbedding(text), nil
	}

	var embeddingResp EmbeddingResponse
	err = json.Unmarshal(body, &embeddingResp)
	if err != nil {
		log.Printf("[Embedding] Failed to unmarshal response: %v", err)
		return generateFallbackEmbedding(text), nil
	}

	log.Printf("[Embedding] Generated %d-dim embedding using %s model", embeddingResp.Dimensions, embeddingResp.Model)
	return embeddingResp.Embedding, nil
}

func generateFallbackEmbedding(text string) []float32 {
	// Simple hash-based fallback embedding (768 dimensions to match all-mpnet-base-v2)
	embedding := make([]float32, 768)
	hash := 0
	for _, char := range text {
		hash = hash*31 + int(char)
	}

	// Generate pseudo-random but deterministic embedding
	for i := 0; i < 768; i++ {
		hash = hash*1103515245 + 12345
		embedding[i] = float32(hash%1000-500) / 1000.0 // Normalize to [-0.5, 0.5]
	}
	return embedding
}

// storeEmbeddingInDB stores embedding in SQLite for semantic search
func storeEmbeddingInDB(sqliteConn *sql.DB, userID string, postIdx int, text string, embedding []float32) error {
	// Convert embedding to JSON for storage
	embeddingJSON, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("failed to marshal embedding: %v", err)
	}

	// Store in SQLite with metadata for semantic search
	postID := fmt.Sprintf("%s_post_%d_%d", userID, postIdx, time.Now().Unix())

	_, err = sqliteConn.Exec(`INSERT OR REPLACE INTO embeddings 
		(post_id, user_id, text, embedding_json, model, dimensions, created_at) 
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		postID, userID, text, string(embeddingJSON), "personality", len(embedding), time.Now().Unix())

	if err != nil {
		return fmt.Errorf("failed to store embedding: %v", err)
	}

	log.Printf("[EmbeddingDB] Stored %d-dim embedding for user %s, post %s", len(embedding), userID, postID)
	return nil
}

func createIngestPostsHandler(sqliteConn *sql.DB) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req IngestRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request payload", "details": err.Error()})
			return
		}

		// Generate production embeddings for semantic search
		embeddingResults := make([][]float32, 0, len(req.Posts))
		successCount := 0

		for i, post := range req.Posts {
			// Extract text content from post (assuming it's a map or string)
			var text string
			if postMap, ok := post.(map[string]interface{}); ok {
				if content, exists := postMap["content"]; exists {
					text = content.(string)
				}
			} else if postStr, ok := post.(string); ok {
				text = postStr
			}

			if text != "" {
				// Generate embedding using personality model for posts
				embedding, err := generateProductionEmbedding(text, "personality")
				if err == nil {
					// Store embedding in database for semantic search
					err = storeEmbeddingInDB(sqliteConn, req.UserID, i, text, embedding)
					if err != nil {
						log.Printf("[Ingestion] Failed to store embedding for post %d: %v", i, err)
					} else {
						embeddingResults = append(embeddingResults, embedding)
						successCount++
					}
				} else {
					log.Printf("[Ingestion] Failed to generate embedding for post %d: %v", i, err)
				}
			}
		}

		log.Printf("[GoBackend] Generated %d/%d embeddings for user %s", successCount, len(req.Posts), req.UserID)
		log.Printf("[GoBackend] Ingesting %d posts for user %s", len(req.Posts), req.UserID)

		c.JSON(http.StatusOK, gin.H{
			"result":           true,
			"embeddings_count": successCount,
			"total_posts":      len(req.Posts),
		})
	}
}

// Add embedding endpoints to the registerEndpoints function
func registerEmbeddingEndpoints(r *gin.Engine, sqliteConn *sql.DB) {
	r.POST("/ingest-posts", createIngestPostsHandler(sqliteConn))

	// Direct embedding endpoint for testing
	r.POST("/generate-embedding", func(c *gin.Context) {
		var req struct {
			Text  string `json:"text"`
			Model string `json:"model"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid payload", "details": err.Error()})
			return
		}

		if req.Text == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Text field is required"})
			return
		}

		if req.Model == "" {
			req.Model = "semantic" // Default model
		}

		embedding, err := generateProductionEmbedding(req.Text, req.Model)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate embedding", "details": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"embedding":   embedding,
			"dimensions":  len(embedding),
			"model":       req.Model,
			"text_length": len(req.Text),
		})
	})
}
