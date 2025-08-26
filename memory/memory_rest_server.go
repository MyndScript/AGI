package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
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

func main() {
	r := gin.Default()
	// Debug endpoint: dump all key-value pairs for a user
	r.POST("/dump_memory", func(c *gin.Context) {
		var req UserContextRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "bad request"})
			return
		}
		mem := memoryStore[req.UserID]
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
			if _, ok := memoryStore[req.UserID]; !ok {
				memoryStore[req.UserID] = make(map[string]string)
			}
			memoryStore[req.UserID][req.Key] = req.Value
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
		if mem, ok := memoryStore[req.UserID]; ok {
			value, found = mem[req.Key]
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
		if mem, ok := memoryStore[req.UserID]; ok {
			for k, v := range mem {
				context += k + ": " + v + "\n"
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
		moments := momentsStore[req.UserID]
		filtered := []Moment{}
		for _, m := range moments {
			if (req.Since == 0 || m.Timestamp >= req.Since) && (req.Until == 0 || m.Timestamp <= req.Until) {
				if len(req.Tags) == 0 {
					filtered = append(filtered, m)
				} else {
					for _, tag := range req.Tags {
						for _, mt := range m.Tags {
							if tag == mt {
								filtered = append(filtered, m)
								break
							}
						}
					}
				}
			}
		}
		c.JSON(http.StatusOK, GetMomentsResponse{Moments: filtered})
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
