package main

import (
	"database/sql"
	"log"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"
)

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

type UserContextResponse struct {
	Context string `json:"context"`
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
type GetMomentsResponse struct {
	Moments []Moment `json:"moments"`
}

type SemanticSearchRequest struct {
	UserID string `json:"user_id"`
	Query  string `json:"query"`
}

type SemanticSearchResponse struct {
	Results []Moment `json:"results"`
}

var memoryStore = make(map[string]map[string]string)
var momentsStore = make(map[string][]Moment)
var userContexts = make(map[string]string)

// Privacy/public split for key-value memory
var privateMemoryStore = make(map[string]map[string]string)
var publicMemoryStore = make(map[string]map[string]string)

// Privacy/public split for moments
var privateMomentsStore = make(map[string][]Moment)
var publicMomentsStore = make(map[string][]Moment)

func main() {
	// Initialize SQLite connection and create table for user memory
	sqliteConn, err := sql.Open("sqlite3", "user_memory.db")
	if err != nil {
		log.Fatalf("Failed to connect to SQLite: %v", err)
	}
	defer sqliteConn.Close()

	// Initialize Postgres connection for global knowledge
	pgConnStr := "postgres://user:password@localhost:5432/global_memory?sslmode=disable" // TODO: update with real credentials
	pgConn, err := sql.Open("postgres", pgConnStr)
	if err != nil {
		log.Fatalf("Failed to connect to Postgres: %v", err)
	}
	defer pgConn.Close()

	_, err = pgConn.Exec(`CREATE TABLE IF NOT EXISTS global_memory (
         user_id TEXT,
         key TEXT,
         value TEXT,
         PRIMARY KEY (user_id, key)
     )`)
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

	r := gin.Default()

	// Endpoint to store global (public) key-value memory in Postgres
	r.POST("/set-global-memory", func(c *gin.Context) {
		var req SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		if req.UserID != "" && req.Key != "" {
			_, err := pgConn.Exec(`INSERT INTO global_memory (user_id, key, value) VALUES ($1, $2, $3) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value`, req.UserID, req.Key, req.Value)
			if err != nil {
				c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
				return
			}
			c.JSON(http.StatusOK, SetResponse{Success: true})
			return
		}
		c.JSON(http.StatusBadRequest, SetResponse{Success: false})
	})

	// Endpoint to get global (public) key-value memory from Postgres
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

	// Endpoint to store global (public) moments in Postgres
	r.POST("/set-global-moment", func(c *gin.Context) {
		var req SetMomentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		m := req.Moment
		_, err := pgConn.Exec(
			`INSERT INTO global_moments (id, user_id, summary, emotion, glyph, tags, timestamp, embedding) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, emotion=excluded.emotion, glyph=excluded.glyph, tags=excluded.tags, timestamp=excluded.timestamp, embedding=excluded.embedding`,
			m.ID, m.UserID, m.Summary, m.Emotion, m.Glyph, strings.Join(m.Tags, ","), m.Timestamp, m.Embedding,
		)
		if err != nil {
			c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
			return
		}
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	// Endpoint to get global (public) moments from Postgres
	r.POST("/get-global-moments", func(c *gin.Context) {
		var req GetMomentsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, GetMomentsResponse{})
			return
		}
		query := `SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM global_moments WHERE user_id = $1`
		args := []interface{}{req.UserID}
		rows, err := pgConn.Query(query, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, GetMomentsResponse{})
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
		c.JSON(http.StatusOK, GetMomentsResponse{Moments: moments})
	})
	// Endpoint to store private/public key-value memory
	r.POST("/set-memory-privacy", func(c *gin.Context) {
		var req SetRequest
		privacy := c.DefaultQuery("privacy", "private")
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		store := privateMemoryStore
		if privacy == "public" {
			store = publicMemoryStore
		}
		if req.UserID != "" && req.Key != "" {
			if _, ok := store[req.UserID]; !ok {
				store[req.UserID] = make(map[string]string)
			}
			store[req.UserID][req.Key] = req.Value
			c.JSON(http.StatusOK, SetResponse{Success: true})
			return
		}
		c.JSON(http.StatusBadRequest, SetResponse{Success: false})
	})

	// Add observation endpoint (compatible with agent)
	r.POST("/add-observation", func(c *gin.Context) {
		var req SetMomentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		m := req.Moment
		_, err := sqliteConn.Exec(
			`INSERT INTO moments (id, user_id, summary, emotion, glyph, tags, timestamp, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			m.ID, m.UserID, m.Summary, m.Emotion, m.Glyph, strings.Join(m.Tags, ","), m.Timestamp, m.Embedding,
		)
		if err != nil {
			c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
			return
		}
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	// Endpoint to store private/public moments
	r.POST("/set_moment_privacy", func(c *gin.Context) {
		var req SetMomentRequest
		privacy := c.DefaultQuery("privacy", "private")
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		store := privateMomentsStore
		if privacy == "public" {
			store = publicMomentsStore
		}
		store[req.Moment.UserID] = append(store[req.Moment.UserID], req.Moment)
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	// Debug endpoint: dump all key-value pairs for a user
	r.POST("/dump_memory", func(c *gin.Context) {
		var req UserContextRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "bad request"})
			return
		}
		mem := make(map[string]string)
		rows, err := sqliteConn.Query(`SELECT key, value FROM user_memory WHERE user_id = ?`, req.UserID)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var k, v string
				if err := rows.Scan(&k, &v); err == nil {
					mem[k] = v
				}
			}
		}
		c.JSON(http.StatusOK, gin.H{"memory": mem})
	})

	// Python-compatible endpoints
	r.POST("/set-memory", func(c *gin.Context) {
		var req SetRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		if req.UserID != "" && req.Key != "" {
			// Upsert into SQLite
			_, err := sqliteConn.Exec(`INSERT INTO user_memory (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value`, req.UserID, req.Key, req.Value)
			if err != nil {
				c.JSON(http.StatusInternalServerError, SetResponse{Success: false})
				return
			}
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

	r.POST("/get-user-context", func(c *gin.Context) {
		var req UserContextRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, UserContextResponse{Context: ""})
			return
		}
		context := ""
		rows, err := sqliteConn.Query(`SELECT key, value FROM user_memory WHERE user_id = ?`, req.UserID)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var k, v string
				if err := rows.Scan(&k, &v); err == nil {
					context += k + ": " + v + "\n"
				}
			}
		}
		c.JSON(http.StatusOK, UserContextResponse{Context: context})
	})

	r.POST("/set_moment", func(c *gin.Context) {
		var req SetMomentRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SetResponse{Success: false})
			return
		}
		momentsStore[req.Moment.UserID] = append(momentsStore[req.Moment.UserID], req.Moment)
		c.JSON(http.StatusOK, SetResponse{Success: true})
	})

	r.POST("/get_moments", func(c *gin.Context) {
		var req GetMomentsRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, GetMomentsResponse{})
			return
		}
		query := `SELECT id, user_id, summary, emotion, glyph, tags, timestamp, embedding FROM moments WHERE user_id = ?`
		args := []interface{}{req.UserID}
		rows, err := sqliteConn.Query(query, args...)
		if err != nil {
			c.JSON(http.StatusInternalServerError, GetMomentsResponse{})
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
		c.JSON(http.StatusOK, GetMomentsResponse{Moments: moments})
	})

	r.POST("/semantic_search", func(c *gin.Context) {
		var req SemanticSearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, SemanticSearchResponse{})
			return
		}
		moments := momentsStore[req.UserID]
		// Dummy search: return all moments for user
		c.JSON(http.StatusOK, SemanticSearchResponse{Results: moments})
	})

	r.Run(":8002")
}
